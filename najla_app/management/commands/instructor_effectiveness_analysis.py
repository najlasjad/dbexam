import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.cluster import KMeans
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from django.core.management.base import BaseCommand
from django.db import connection
import warnings
warnings.filterwarnings('ignore')

class Command(BaseCommand):
    help = 'Instructor Effectiveness Analysis - Analyze instructor impact on student performance'

    def add_arguments(self, parser):
        parser.add_argument('--action', type=str, choices=['train', 'test', 'predict', 'analyze', 'compare'], 
                            default='analyze', help='Action to perform')
        parser.add_argument('--instructor_id', type=int, help='Instructor ID for analysis')
        parser.add_argument('--course_id', type=int, help='Course ID for prediction')
        parser.add_argument('--semester_id', type=int, help='Semester ID for prediction')

    def handle(self, *args, **options):
        if options['action'] == 'train':
            self.train_model()
        elif options['action'] == 'test':
            self.test_model()
        elif options['action'] == 'predict':
            self.predict_instructor_impact(options['instructor_id'], options['course_id'], options['semester_id'])
        elif options['action'] == 'analyze':
            self.analyze_instructor_effectiveness()
        elif options['action'] == 'compare':
            self.compare_instructors()

    def get_instructor_data(self):
        """Extract comprehensive instructor performance data"""
        query = """
        SELECT 
            ci.instructor_id,
            i.instructor_name,
            ci.course_id,
            c.course_name,
            ci.semester_id,
            sem.semester_name,
            e.stu_id,
            s.name as student_name,
            e.grade,
            s.gender,
            d.dept_name,
            COALESCE(cd.difficulty_level, 'Medium') as difficulty_level,
            COALESCE(att.attendance_percentage, 75) as attendance,
            COALESCE(avg_assess.avg_assessment, 75) as avg_assessment_score,
            -- Instructor historical metrics
            COALESCE(inst_stats.avg_class_grade, 75) as instructor_avg_grade,
            COALESCE(inst_stats.class_count, 1) as instructor_class_count,
            COALESCE(inst_stats.student_count, 0) as instructor_student_count
        FROM course_instructor ci
        JOIN instructor i ON ci.instructor_id = i.instructor_id
        JOIN course c ON ci.course_id = c.course_id
        JOIN semester sem ON ci.semester_id = sem.semester_id
        JOIN enrollment e ON (e.course_id = ci.course_id AND e.semester_id = ci.semester_id)
        JOIN student s ON e.stu_id = s.stu_id
        JOIN department d ON s.dept_id = d.dept_id
        LEFT JOIN course_difficulty cd ON c.course_id = cd.course_id
        LEFT JOIN attendance att ON e.enroll_id = att.enroll_id
        LEFT JOIN (
            SELECT enroll_id, AVG(score) as avg_assessment
            FROM assessment
            GROUP BY enroll_id
        ) avg_assess ON e.enroll_id = avg_assess.enroll_id
        LEFT JOIN (
            SELECT 
                ci2.instructor_id,
                AVG(e2.grade) as avg_class_grade,
                COUNT(DISTINCT CONCAT(ci2.course_id, '-', ci2.semester_id)) as class_count,
                COUNT(DISTINCT e2.stu_id) as student_count
            FROM course_instructor ci2
            JOIN enrollment e2 ON (e2.course_id = ci2.course_id AND e2.semester_id = ci2.semester_id)
            GROUP BY ci2.instructor_id
        ) inst_stats ON ci.instructor_id = inst_stats.instructor_id
        ORDER BY ci.instructor_id, ci.course_id, ci.semester_id, e.stu_id
        """
        
        with connection.cursor() as cursor:
            cursor.execute(query)
            data = cursor.fetchall()
        
        columns = ['instructor_id', 'instructor_name', 'course_id', 'course_name', 
                    'semester_id', 'semester_name', 'stu_id', 'student_name', 'grade',
                    'gender', 'dept_name', 'difficulty_level', 'attendance', 
                    'avg_assessment_score', 'instructor_avg_grade', 'instructor_class_count',
                    'instructor_student_count']
        
        df = pd.DataFrame(data, columns=columns)
        return df

    def prepare_instructor_features(self, df):
        """Prepare features for instructor effectiveness analysis"""
        # Encode categorical variables
        le_gender = LabelEncoder()
        le_dept = LabelEncoder()
        le_difficulty = LabelEncoder()
        
        df_encoded = df.copy()
        df_encoded['gender_encoded'] = le_gender.fit_transform(df['gender'])
        df_encoded['dept_encoded'] = le_dept.fit_transform(df['dept_name'])
        df_encoded['difficulty_encoded'] = le_difficulty.fit_transform(df['difficulty_level'])
        
        # Calculate instructor-specific metrics
        instructor_metrics = df_encoded.groupby('instructor_id').agg({
            'grade': ['mean', 'std', 'count'],
            'attendance': 'mean',
            'avg_assessment_score': 'mean'
        }).reset_index()
        
        instructor_metrics.columns = ['instructor_id', 'instructor_grade_mean', 'instructor_grade_std', 
                                    'total_students', 'avg_attendance_supervised', 'avg_assessment_supervised']
        instructor_metrics['instructor_grade_std'] = instructor_metrics['instructor_grade_std'].fillna(0)
        
        df_encoded = df_encoded.merge(instructor_metrics, on='instructor_id')
        
        # Calculate course-level baselines (performance without specific instructor)
        course_baselines = df_encoded.groupby('course_id').agg({
            'grade': 'mean',
            'attendance': 'mean'
        }).reset_index()
        course_baselines.columns = ['course_id', 'course_baseline_grade', 'course_baseline_attendance']
        
        df_encoded = df_encoded.merge(course_baselines, on='course_id')
        
        # Calculate instructor effectiveness metrics
        df_encoded['grade_difference'] = df_encoded['grade'] - df_encoded['course_baseline_grade']
        df_encoded['attendance_impact'] = df_encoded['attendance'] - df_encoded['course_baseline_attendance']
        
        # Feature columns
        feature_columns = ['instructor_id', 'course_id', 'semester_id', 'gender_encoded', 
                            'dept_encoded', 'difficulty_encoded', 'attendance', 'avg_assessment_score',
                            'instructor_grade_mean', 'instructor_grade_std', 'total_students',
                            'course_baseline_grade', 'course_baseline_attendance']
        
        X = df_encoded[feature_columns].fillna(df_encoded[feature_columns].mean())
        y = df_encoded['grade']
        
        # Save encoders
        encoders = {
            'gender': le_gender,
            'dept': le_dept,
            'difficulty': le_difficulty
        }
        
        joblib.dump(encoders, 'instructor_effectiveness_encoders.pkl')
        
        return X, y, df_encoded, feature_columns

    def train_model(self):
        """Train the instructor effectiveness prediction model"""
        self.stdout.write("Loading instructor data...")
        df = self.get_instructor_data()
        
        self.stdout.write("Preparing features...")
        X, y, df_encoded, feature_columns = self.prepare_instructor_features(df)
        
        self.stdout.write("Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        self.stdout.write("Training model...")
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Save model and scaler
        joblib.dump(model, 'instructor_effectiveness_model.pkl')
        joblib.dump(scaler, 'instructor_effectiveness_scaler.pkl')
        joblib.dump(feature_columns, 'instructor_effectiveness_features.pkl')
        
        # Evaluate
        y_pred = model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        self.stdout.write(f"Model trained successfully!")
        self.stdout.write(f"R² Score: {r2:.4f}")
        self.stdout.write(f"Mean Squared Error: {mse:.4f}")
        self.stdout.write(f"Mean Absolute Error: {mae:.4f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        self.stdout.write("Feature Importance:")
        self.stdout.write(str(feature_importance.head(10)))

    def test_model(self):
        """Test the trained model with comprehensive evaluation"""
        # Load model
        model = joblib.load('instructor_effectiveness_model.pkl')
        scaler = joblib.load('instructor_effectiveness_scaler.pkl')
        feature_columns = joblib.load('instructor_effectiveness_features.pkl')
        
        # Get data
        df = self.get_instructor_data()
        X, y, df_encoded, _ = self.prepare_instructor_features(df)
        
        # Scale and predict
        X_scaled = scaler.transform(X)
        y_pred = model.predict(X_scaled)
        
        # Metrics
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        
        self.stdout.write(f"Test Results:")
        self.stdout.write(f"R² Score: {r2:.4f}")
        self.stdout.write(f"Mean Squared Error: {mse:.4f}")
        self.stdout.write(f"Mean Absolute Error: {mae:.4f}")
        
        # Generate visualizations
        self.generate_effectiveness_visualizations(y, y_pred, df_encoded, model, feature_columns)

    def predict_instructor_impact(self, instructor_id, course_id, semester_id):
        """Predict instructor impact on student performance"""
        if not all([instructor_id, course_id, semester_id]):
            self.stdout.write("Please provide instructor_id, course_id, and semester_id")
            return
        
        # Load model
        model = joblib.load('instructor_effectiveness_model.pkl')
        scaler = joblib.load('instructor_effectiveness_scaler.pkl')
        encoders = joblib.load('instructor_effectiveness_encoders.pkl')
        
        # Get instructor and course data
        df = self.get_instructor_data()
        
        # Filter for specific instructor-course-semester combination
        specific_data = df[(df['instructor_id'] == instructor_id) & 
                            (df['course_id'] == course_id) & 
                            (df['semester_id'] == semester_id)]
        
        if specific_data.empty:
            self.stdout.write("No data found for the specified combination")
            return
        
        # Prepare features
        X, y, df_encoded, _ = self.prepare_instructor_features(df)
        
        # Get prediction for this combination
        specific_encoded = df_encoded[(df_encoded['instructor_id'] == instructor_id) & 
                                    (df_encoded['course_id'] == course_id) & 
                                    (df_encoded['semester_id'] == semester_id)]
        
        if specific_encoded.empty:
            self.stdout.write("No encoded data found")
            return
        
        # Calculate average predicted performance
        X_specific = specific_encoded[X.columns]
        X_scaled = scaler.transform(X_specific)
        predictions = model.predict(X_scaled)
        
        avg_predicted = predictions.mean()
        actual_avg = specific_encoded['grade'].mean()
        baseline_avg = specific_encoded['course_baseline_grade'].mean()
        
        effectiveness_score = avg_predicted - baseline_avg
        actual_effectiveness = actual_avg - baseline_avg
        
        self.stdout.write(f"Instructor Effectiveness Analysis:")
        self.stdout.write(f"Instructor: {specific_data['instructor_name'].iloc[0]}")
        self.stdout.write(f"Course: {specific_data['course_name'].iloc[0]}")
        self.stdout.write(f"Semester: {specific_data['semester_name'].iloc[0]}")
        self.stdout.write(f"")
        self.stdout.write(f"Course Baseline Average: {baseline_avg:.2f}")
        self.stdout.write(f"Predicted Average Grade: {avg_predicted:.2f}")
        self.stdout.write(f"Actual Average Grade: {actual_avg:.2f}")
        self.stdout.write(f"Predicted Effectiveness: {effectiveness_score:+.2f}")
        self.stdout.write(f"Actual Effectiveness: {actual_effectiveness:+.2f}")
        
        # Performance assessment
        if effectiveness_score > 2:
            assessment = "Highly Effective"
        elif effectiveness_score > 0:
            assessment = "Effective"
        elif effectiveness_score > -2:
            assessment = "Average"
        else:
            assessment = "Needs Improvement"
            
        self.stdout.write(f"Performance Assessment: {assessment}")

    def analyze_instructor_effectiveness(self):
        """Comprehensive instructor effectiveness analysis"""
        df = self.get_instructor_data()
        
        self.stdout.write("=== INSTRUCTOR EFFECTIVENESS ANALYSIS ===")
        
        # Overall statistics
        instructor_summary = df.groupby(['instructor_id', 'instructor_name']).agg({
            'grade': ['mean', 'std', 'count'],
            'attendance': 'mean',
            'avg_assessment_score': 'mean',
            'course_id': 'nunique',
            'semester_id': 'nunique'
        }).round(2)
        
        instructor_summary.columns = ['avg_grade', 'grade_std', 'total_students', 
                                    'avg_attendance', 'avg_assessment', 'courses_taught', 'semesters_taught']
        
        # Calculate effectiveness metrics
        course_averages = df.groupby('course_id')['grade'].mean()
        df_with_baseline = df.merge(course_averages.rename('course_baseline'), on='course_id')
        df_with_baseline['effectiveness'] = df_with_baseline['grade'] - df_with_baseline['course_baseline']
        
        effectiveness_summary = df_with_baseline.groupby(['instructor_id', 'instructor_name']).agg({
            'effectiveness': 'mean'
        }).round(2)
        
        final_summary = instructor_summary.merge(effectiveness_summary, left_index=True, right_index=True)
        final_summary = final_summary.sort_values('effectiveness', ascending=False)
        
        self.stdout.write("Top 10 Most Effective Instructors:")
        self.stdout.write(str(final_summary.head(10)))
        
        self.stdout.write("\nBottom 5 Instructors (Need Support):")
        self.stdout.write(str(final_summary.tail(5)))
        
        # Generate analysis visualizations
        self.generate_instructor_analysis_plots(df, final_summary)

    def compare_instructors(self):
        """Compare instructors using clustering analysis"""
        df = self.get_instructor_data()
        
        # Prepare instructor comparison metrics
        instructor_metrics = df.groupby('instructor_id').agg({
            'grade': ['mean', 'std'],
            'attendance': 'mean',
            'avg_assessment_score': 'mean',
            'stu_id': 'nunique',
            'course_id': 'nunique'
        }).round(2)
        
        instructor_metrics.columns = ['avg_grade', 'grade_consistency', 'avg_attendance', 
                                    'avg_assessment', 'students_taught', 'courses_taught']
        
        # Add instructor names
        instructor_names = df.groupby('instructor_id')['instructor_name'].first()
        instructor_metrics = instructor_metrics.merge(instructor_names, left_index=True, right_index=True)
        
        # Clustering analysis
        features_for_clustering = ['avg_grade', 'grade_consistency', 'avg_attendance', 
                                 'avg_assessment', 'students_taught', 'courses_taught']
        
        # Scale features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(instructor_metrics[features_for_clustering])
        
        # Perform clustering
        kmeans = KMeans(n_clusters=4, random_state=42)
        instructor_metrics['cluster'] = kmeans.fit_predict(scaled_features)
        
        # Analyze clusters
        cluster_analysis = instructor_metrics.groupby('cluster').agg({
            'avg_grade': 'mean',
            'grade_consistency': 'mean',
            'avg_attendance': 'mean',
            'avg_assessment': 'mean',
            'students_taught': 'mean',
            'courses_taught': 'mean',
            'instructor_name': 'count'
        }).round(2)
        
        cluster_analysis.columns = ['avg_grade', 'grade_consistency', 'avg_attendance', 
                                  'avg_assessment', 'avg_students', 'avg_courses', 'instructor_count']
        
        self.stdout.write("=== INSTRUCTOR CLUSTERING ANALYSIS ===")
        self.stdout.write(str(cluster_analysis))
        
        # Label clusters
        cluster_labels = {
            cluster_analysis['avg_grade'].idxmax(): "High Performers",
            cluster_analysis['grade_consistency'].idxmin(): "Consistent Educators", 
            cluster_analysis['students_taught'].idxmax(): "High Volume Instructors",
            cluster_analysis['avg_grade'].idxmin(): "Development Needed"
        }
        
        for cluster_id, label in cluster_labels.items():
            instructors_in_cluster = instructor_metrics[instructor_metrics['cluster'] == cluster_id]['instructor_name'].tolist()
            self.stdout.write(f"\n{label} (Cluster {cluster_id}):")
            self.stdout.write(f"Instructors: {', '.join(instructors_in_cluster[:5])}{'...' if len(instructors_in_cluster) > 5 else ''}")

    def generate_effectiveness_visualizations(self, y_true, y_pred, df_encoded, model, feature_columns):
        """Generate instructor effectiveness visualizations"""
        plt.style.use('default')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Model performance
        axes[0,0].scatter(y_true, y_pred, alpha=0.6)
        axes[0,0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        axes[0,0].set_xlabel('Actual Grades')
        axes[0,0].set_ylabel('Predicted Grades')
        axes[0,0].set_title('Model Performance: Actual vs Predicted')
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=True).tail(10)
        
        axes[0,1].barh(feature_importance['feature'], feature_importance['importance'])
        axes[0,1].set_title('Top 10 Feature Importance')
        axes[0,1].set_xlabel('Importance')
        
        # Instructor effectiveness distribution
        instructor_effect = df_encoded.groupby('instructor_id')['grade_difference'].mean()
        axes[0,2].hist(instructor_effect, bins=20, edgecolor='black', alpha=0.7)
        axes[0,2].set_title('Instructor Effectiveness Distribution')
        axes[0,2].set_xlabel('Grade Difference from Baseline')
        axes[0,2].set_ylabel('Number of Instructors')
        
        # Grade vs attendance by instructor effectiveness
        high_eff = df_encoded[df_encoded['grade_difference'] > 2]
        low_eff = df_encoded[df_encoded['grade_difference'] < -2]
        
        axes[1,0].scatter(high_eff['attendance'], high_eff['grade'], alpha=0.6, color='green', label='High Effectiveness')
        axes[1,0].scatter(low_eff['attendance'], low_eff['grade'], alpha=0.6, color='red', label='Low Effectiveness')
        axes[1,0].set_xlabel('Attendance %')
        axes[1,0].set_ylabel('Grade')
        axes[1,0].set_title('Attendance vs Grade by Instructor Effectiveness')
        axes[1,0].legend()
        
        # Course difficulty impact
        difficulty_impact = df_encoded.groupby(['difficulty_level', 'instructor_id'])['grade'].mean().reset_index()
        difficulty_avg = difficulty_impact.groupby('difficulty_level')['grade'].mean()
        
        axes[1,1].bar(difficulty_avg.index, difficulty_avg.values, color=['green', 'orange', 'red'])
        axes[1,1].set_title('Average Grade by Course Difficulty')
        axes[1,1].set_ylabel('Average Grade')
        
        # Instructor workload vs performance
        workload_perf = df_encoded.groupby('instructor_id').agg({
            'total_students': 'first',
            'grade': 'mean'
        })
        
        axes[1,2].scatter(workload_perf['total_students'], workload_perf['grade'], alpha=0.7)
        axes[1,2].set_xlabel('Total Students Taught')
        axes[1,2].set_ylabel('Average Grade Given')
        axes[1,2].set_title('Instructor Workload vs Performance')
        
        plt.tight_layout()
        plt.savefig('instructor_effectiveness_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

    def generate_instructor_analysis_plots(self, df, summary):
        """Generate comprehensive instructor analysis plots"""
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Top performers
        top_10 = summary.head(10)
        axes[0,0].barh(range(len(top_10)), top_10['effectiveness'])
        axes[0,0].set_yticks(range(len(top_10)))
        axes[0,0].set_yticklabels([name.split()[-1] for name in top_10.index.get_level_values(1)])
        axes[0,0].set_xlabel('Effectiveness Score')
        axes[0,0].set_title('Top 10 Most Effective Instructors')
        
        # Grade distribution by effectiveness
        effectiveness_categories = pd.cut(summary['effectiveness'], bins=[-np.inf, -1, 1, np.inf], 
                                        labels=['Low', 'Average', 'High'])
        effectiveness_counts = effectiveness_categories.value_counts()
        
        axes[0,1].pie(effectiveness_counts.values, labels=effectiveness_counts.index, autopct='%1.1f%%')
        axes[0,1].set_title('Distribution of Instructor Effectiveness')
        
        # Students taught vs effectiveness
        axes[1,0].scatter(summary['total_students'], summary['effectiveness'], alpha=0.7)
        axes[1,0].set_xlabel('Total Students Taught')
        axes[1,0].set_ylabel('Effectiveness Score')
        axes[1,0].set_title('Teaching Load vs Effectiveness')
        
        # Grade consistency vs average grade
        axes[1,1].scatter(summary['grade_std'], summary['avg_grade'], alpha=0.7)
        axes[1,1].set_xlabel('Grade Standard Deviation (Consistency)')
        axes[1,1].set_ylabel('Average Grade')
        axes[1,1].set_title('Grading Consistency vs Average Grade')
        
        plt.tight_layout()
        plt.savefig('instructor_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        self.stdout.write("Comprehensive analysis visualizations saved as 'instructor_comprehensive_analysis.png'")