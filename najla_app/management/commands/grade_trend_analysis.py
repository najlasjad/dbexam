# najla_app/management/commands/grade_trend_analysis.py

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from django.core.management.base import BaseCommand
from django.db import connection
import warnings
warnings.filterwarnings('ignore')

class Command(BaseCommand):
    help = 'Grade Trend Analysis - Analyze and predict grade trends across semesters'

    def add_arguments(self, parser):
        parser.add_argument('--action', type=str, choices=['train', 'test', 'predict', 'analyze'], 
                            default='analyze', help='Action to perform')
        parser.add_argument('--student_id', type=int, help='Student ID for prediction')
        parser.add_argument('--course_id', type=int, help='Course ID for prediction')
        parser.add_argument('--future_semesters', type=int, default=2, help='Number of future semesters to predict')

    def handle(self, *args, **options):
        if options['action'] == 'train':
            self.train_model()
        elif options['action'] == 'test':
            self.test_model()
        elif options['action'] == 'predict':
            self.predict_future_grades(options['student_id'], options['course_id'], options['future_semesters'])
        elif options['action'] == 'analyze':
            self.analyze_trends()

    def get_grade_data(self):
        """Extract grade data with temporal information"""
        query = """
        SELECT 
            e.stu_id,
            e.course_id,
            e.semester_id,
            e.grade,
            s.name as student_name,
            c.course_name,
            sem.semester_name,
            d.dept_name,
            COALESCE(cd.difficulty_level, 'Medium') as difficulty_level,
            COALESCE(att.attendance_percentage, 75) as attendance,
            COALESCE(avg_assess.avg_assessment, 75) as avg_assessment_score,
            -- Semester ordering (assuming semester_id represents chronological order)
            e.semester_id as semester_order,
            -- Previous semester grade for the same student
            LAG(e.grade) OVER (PARTITION BY e.stu_id ORDER BY e.semester_id) as prev_grade,
            -- Average grade in previous semester
            LAG(AVG(e.grade) OVER (PARTITION BY e.stu_id, e.semester_id)) OVER (PARTITION BY e.stu_id ORDER BY e.semester_id) as prev_semester_avg
        FROM enrollment e
        JOIN student s ON e.stu_id = s.stu_id
        JOIN course c ON e.course_id = c.course_id
        JOIN semester sem ON e.semester_id = sem.semester_id
        JOIN department d ON s.dept_id = d.dept_id
        LEFT JOIN course_difficulty cd ON c.course_id = cd.course_id
        LEFT JOIN attendance att ON e.enroll_id = att.enroll_id
        LEFT JOIN (
            SELECT enroll_id, AVG(score) as avg_assessment
            FROM assessment
            GROUP BY enroll_id
        ) avg_assess ON e.enroll_id = avg_assess.enroll_id
        ORDER BY e.stu_id, e.semester_id
        """
        
        with connection.cursor() as cursor:
            cursor.execute(query)
            data = cursor.fetchall()
        
        columns = ['stu_id', 'course_id', 'semester_id', 'grade', 'student_name', 
                    'course_name', 'semester_name', 'dept_name', 'difficulty_level',
                    'attendance', 'avg_assessment_score', 'semester_order', 
                    'prev_grade', 'prev_semester_avg']
        
        df = pd.DataFrame(data, columns=columns)
        return df

    def prepare_trend_features(self, df):
        """Prepare features for trend analysis"""
        # Remove rows with missing previous grades (first semester students)
        df_clean = df.dropna(subset=['prev_grade']).copy()
        
        # Encode difficulty level
        difficulty_map = {'Easy': 1, 'Medium': 2, 'Hard': 3}
        df_clean['difficulty_numeric'] = df_clean['difficulty_level'].map(difficulty_map)
        
        # Calculate trend features
        df_clean['grade_change'] = df_clean['grade'] - df_clean['prev_grade']
        df_clean['grade_improvement'] = (df_clean['grade_change'] > 0).astype(int)
        
        # Student historical performance features
        student_stats = df_clean.groupby('stu_id').agg({
            'grade': ['mean', 'std', 'count'],
            'attendance': 'mean',
            'avg_assessment_score': 'mean'
        }).reset_index()
        
        student_stats.columns = ['stu_id', 'historical_grade_mean', 'historical_grade_std', 
                                'course_count', 'avg_attendance', 'avg_assessment']
        student_stats['historical_grade_std'] = student_stats['historical_grade_std'].fillna(0)
        
        df_clean = df_clean.merge(student_stats, on='stu_id')
        
        # Feature columns for prediction
        feature_columns = ['semester_order', 'prev_grade', 'prev_semester_avg', 
                            'difficulty_numeric', 'attendance', 'avg_assessment_score',
                            'historical_grade_mean', 'historical_grade_std', 'course_count']
        
        X = df_clean[feature_columns].fillna(df_clean[feature_columns].mean())
        y = df_clean['grade']
        
        return X, y, df_clean, feature_columns

    def train_model(self):
        """Train the grade prediction model"""
        self.stdout.write("Loading grade data...")
        df = self.get_grade_data()
        
        self.stdout.write("Preparing features...")
        X, y, df_clean, feature_columns = self.prepare_trend_features(df)
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train model
        self.stdout.write("Training model...")
        model = LinearRegression()
        model.fit(X_scaled, y)
        
        # Save model and scaler
        joblib.dump(model, 'grade_trend_model.pkl')
        joblib.dump(scaler, 'grade_trend_scaler.pkl')
        joblib.dump(feature_columns, 'grade_trend_features.pkl')
        
        # Evaluate
        y_pred = model.predict(X_scaled)
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        
        self.stdout.write(f"Model trained successfully!")
        self.stdout.write(f"R² Score: {r2:.4f}")
        self.stdout.write(f"Mean Squared Error: {mse:.4f}")
        self.stdout.write(f"Mean Absolute Error: {mae:.4f}")
        
        # Feature importance (coefficients)
        feature_importance = pd.DataFrame({
            'feature': feature_columns,
            'coefficient': model.coef_
        }).sort_values('coefficient', key=abs, ascending=False)
        
        self.stdout.write("Feature Coefficients:")
        self.stdout.write(str(feature_importance))

    def test_model(self):
        """Test the trained model with visualizations"""
        # Load model
        model = joblib.load('grade_trend_model.pkl')
        scaler = joblib.load('grade_trend_scaler.pkl')
        feature_columns = joblib.load('grade_trend_features.pkl')
        
        # Get data
        df = self.get_grade_data()
        X, y, df_clean, _ = self.prepare_trend_features(df)
        
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
        self.generate_trend_visualizations(y, y_pred, df_clean, model, feature_columns)

    def predict_future_grades(self, student_id, course_id, future_semesters):
        """Predict future grades for a specific student and course"""
        if not student_id:
            self.stdout.write("Please provide student_id")
            return
        
        # Load model
        model = joblib.load('grade_trend_model.pkl')
        scaler = joblib.load('grade_trend_scaler.pkl')
        
        # Get student's historical data
        df = self.get_grade_data()
        student_data = df[df['stu_id'] == student_id].copy()
        
        if student_data.empty:
            self.stdout.write(f"No data found for student {student_id}")
            return
        
        # Get latest semester data
        latest_semester = student_data['semester_order'].max()
        latest_grade = student_data[student_data['semester_order'] == latest_semester]['grade'].iloc[0]
        
        # Prepare base features
        X, y, df_clean, _ = self.prepare_trend_features(df)
        student_features = df_clean[df_clean['stu_id'] == student_id].iloc[-1].copy()
        
        predictions = []
        current_grade = latest_grade
        
        for i in range(1, future_semesters + 1):
            # Update features for prediction
            feature_values = [
                latest_semester + i,  # semester_order
                current_grade,        # prev_grade
                current_grade,        # prev_semester_avg (simplified)
                student_features['difficulty_numeric'],
                student_features['attendance'],
                student_features['avg_assessment_score'],
                student_features['historical_grade_mean'],
                student_features['historical_grade_std'],
                student_features['course_count']
            ]
            
            # Scale and predict
            X_pred = scaler.transform([feature_values])
            predicted_grade = model.predict(X_pred)[0]
            
            predictions.append({
                'semester': latest_semester + i,
                'predicted_grade': predicted_grade,
                'grade_change': predicted_grade - current_grade
            })
            
            current_grade = predicted_grade
        
        # Display results
        self.stdout.write(f"Grade Predictions for Student {student_id}:")
        self.stdout.write(f"Current Grade (Semester {latest_semester}): {latest_grade:.2f}")
        self.stdout.write("Future Predictions:")
        
        for pred in predictions:
            change_indicator = "↑" if pred['grade_change'] > 0 else "↓" if pred['grade_change'] < 0 else "→"
            self.stdout.write(f"Semester {pred['semester']}: {pred['predicted_grade']:.2f} "
                            f"({change_indicator} {pred['grade_change']:+.2f})")

    def analyze_trends(self):
        """Analyze overall grade trends"""
        df = self.get_grade_data()
        
        # Overall statistics
        self.stdout.write("=== GRADE TREND ANALYSIS ===")
        self.stdout.write(f"Total records: {len(df)}")
        self.stdout.write(f"Average grade: {df['grade'].mean():.2f}")
        self.stdout.write(f"Grade standard deviation: {df['grade'].std():.2f}")
        
        # Trend by semester
        semester_trends = df.groupby('semester_id').agg({
            'grade': ['mean', 'std', 'count']
        }).round(2)
        semester_trends.columns = ['avg_grade', 'std_grade', 'student_count']
        
        self.stdout.write("\n=== SEMESTER TRENDS ===")
        self.stdout.write(str(semester_trends))
        
        # Department trends
        dept_trends = df.groupby('dept_name').agg({
            'grade': ['mean', 'std', 'count']
        }).round(2)
        dept_trends.columns = ['avg_grade', 'std_grade', 'student_count']
        
        self.stdout.write("\n=== DEPARTMENT TRENDS ===")
        self.stdout.write(str(dept_trends))
        
        # Course difficulty impact
        difficulty_impact = df.groupby('difficulty_level').agg({
            'grade': ['mean', 'std', 'count']
        }).round(2)
        difficulty_impact.columns = ['avg_grade', 'std_grade', 'course_count']
        
        self.stdout.write("\n=== DIFFICULTY LEVEL IMPACT ===")
        self.stdout.write(str(difficulty_impact))
        
        # Generate trend visualizations
        self.generate_trend_analysis_plots(df)

    def generate_trend_visualizations(self, y_true, y_pred, df_clean, model, feature_columns):
        """Generate visualization plots for model performance"""
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Actual vs Predicted
        axes[0,0].scatter(y_true, y_pred, alpha=0.6, color='blue')
        axes[0,0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        axes[0,0].set_xlabel('Actual Grades')
        axes[0,0].set_ylabel('Predicted Grades')
        axes[0,0].set_title('Actual vs Predicted Grades')
        
        # Residuals plot
        residuals = y_true - y_pred
        axes[0,1].scatter(y_pred, residuals, alpha=0.6, color='green')
        axes[0,1].axhline(y=0, color='r', linestyle='--')
        axes[0,1].set_xlabel('Predicted Grades')
        axes[0,1].set_ylabel('Residuals')
        axes[0,1].set_title('Residual Plot')
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_columns,
            'coefficient': abs(model.coef_)
        }).sort_values('coefficient', ascending=True)
        
        axes[1,0].barh(feature_importance['feature'], feature_importance['coefficient'])
        axes[1,0].set_title('Feature Importance (Absolute Coefficients)')
        axes[1,0].set_xlabel('Absolute Coefficient Value')
        
        # Grade distribution
        axes[1,1].hist(y_true, bins=30, alpha=0.7, label='Actual', color='blue')
        axes[1,1].hist(y_pred, bins=30, alpha=0.7, label='Predicted', color='red')
        axes[1,1].set_xlabel('Grade')
        axes[1,1].set_ylabel('Frequency')
        axes[1,1].set_title('Grade Distribution Comparison')
        axes[1,1].legend()
        
        plt.tight_layout()
        plt.savefig('grade_trend_model_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

    def generate_trend_analysis_plots(self, df):
        """Generate trend analysis visualizations"""
        plt.style.use('default')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Semester trend
        semester_avg = df.groupby('semester_id')['grade'].mean()
        axes[0,0].plot(semester_avg.index, semester_avg.values, marker='o')
        axes[0,0].set_title('Average Grade by Semester')
        axes[0,0].set_xlabel('Semester ID')
        axes[0,0].set_ylabel('Average Grade')
        axes[0,0].grid(True)
        
        # Department comparison
        dept_data = df.groupby('dept_name')['grade'].mean().sort_values(ascending=False)
        axes[0,1].bar(range(len(dept_data)), dept_data.values)
        axes[0,1].set_title('Average Grade by Department')
        axes[0,1].set_ylabel('Average Grade')
        axes[0,1].set_xticks(range(len(dept_data)))
        axes[0,1].set_xticklabels(dept_data.index, rotation=45)
        
        # Difficulty level impact
        difficulty_data = df.groupby('difficulty_level')['grade'].mean()
        axes[0,2].bar(difficulty_data.index, difficulty_data.values, 
                        color=['green', 'orange', 'red'])
        axes[0,2].set_title('Average Grade by Course Difficulty')
        axes[0,2].set_ylabel('Average Grade')
        
        # Grade distribution
        axes[1,0].hist(df['grade'], bins=30, edgecolor='black', alpha=0.7)
        axes[1,0].set_title('Overall Grade Distribution')
        axes[1,0].set_xlabel('Grade')
        axes[1,0].set_ylabel('Frequency')
        
        # Attendance vs Grade correlation
        valid_attendance = df.dropna(subset=['attendance', 'grade'])
        axes[1,1].scatter(valid_attendance['attendance'], valid_attendance['grade'], alpha=0.6)
        axes[1,1].set_xlabel('Attendance %')
        axes[1,1].set_ylabel('Grade')
        axes[1,1].set_title('Attendance vs Grade Relationship')
        
        # Assessment vs Grade correlation
        valid_assessment = df.dropna(subset=['avg_assessment_score', 'grade'])
        axes[1,2].scatter(valid_assessment['avg_assessment_score'], valid_assessment['grade'], alpha=0.6)
        axes[1,2].set_xlabel('Average Assessment Score')
        axes[1,2].set_ylabel('Grade')
        axes[1,2].set_title('Assessment vs Grade Relationship')
        
        plt.tight_layout()
        plt.savefig('grade_trend_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        self.stdout.write("Trend analysis visualizations saved as 'grade_trend_analysis.png'")