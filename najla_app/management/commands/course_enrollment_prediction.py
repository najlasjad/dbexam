# najla_app/management/commands/course_enrollment_prediction.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from django.core.management.base import BaseCommand
from django.db import connection
import os

class Command(BaseCommand):
    help = 'Course Enrollment Prediction - Predict whether a student will enroll in a course'

    def add_arguments(self, parser):
        parser.add_argument('--action', type=str, choices=['train', 'test', 'predict'], 
                            default='train', help='Action to perform')
        parser.add_argument('--student_id', type=int, help='Student ID for prediction')
        parser.add_argument('--course_id', type=int, help='Course ID for prediction')
        parser.add_argument('--semester_id', type=int, help='Semester ID for prediction')

    def handle(self, *args, **options):
        if options['action'] == 'train':
            self.train_model()
        elif options['action'] == 'test':
            self.test_model()
        elif options['action'] == 'predict':
            self.predict_enrollment(options['student_id'], options['course_id'], options['semester_id'])

    def get_data(self):
        """Extract enrollment data with features"""
        query = """
        SELECT 
            e.stu_id,
            e.course_id,
            e.semester_id,
            s.gender,
            d.dept_name as student_dept,
            cd.dept_name as course_dept,
            CASE WHEN cd_diff.difficulty_level IS NOT NULL THEN cd_diff.difficulty_level ELSE 'Medium' END as difficulty_level,
            COALESCE(avg_attendance.avg_attendance, 75) as historical_attendance,
            COALESCE(avg_grade.avg_grade, 75) as historical_grade,
            1 as enrolled
        FROM enrollment e
        JOIN student s ON e.stu_id = s.stu_id
        JOIN course c ON e.course_id = c.course_id
        JOIN department d ON s.dept_id = d.dept_id
        JOIN department cd ON c.dept_id = cd.dept_id
        LEFT JOIN course_difficulty cd_diff ON c.course_id = cd_diff.course_id
        LEFT JOIN (
            SELECT stu_id, AVG(attendance_percentage) as avg_attendance
            FROM attendance a
            JOIN enrollment e ON a.enroll_id = e.enroll_id
            GROUP BY stu_id
        ) avg_attendance ON e.stu_id = avg_attendance.stu_id
        LEFT JOIN (
            SELECT stu_id, AVG(grade) as avg_grade
            FROM enrollment
            GROUP BY stu_id
        ) avg_grade ON e.stu_id = avg_grade.stu_id
        """
        
        # Create negative samples (students who didn't enroll)
        negative_query = """
        SELECT DISTINCT
            s.stu_id,
            c.course_id,
            sem.semester_id,
            s.gender,
            d.dept_name as student_dept,
            cd.dept_name as course_dept,
            CASE WHEN cd_diff.difficulty_level IS NOT NULL THEN cd_diff.difficulty_level ELSE 'Medium' END as difficulty_level,
            COALESCE(avg_attendance.avg_attendance, 75) as historical_attendance,
            COALESCE(avg_grade.avg_grade, 75) as historical_grade,
            0 as enrolled
        FROM student s
        CROSS JOIN course c
        CROSS JOIN semester sem
        JOIN department d ON s.dept_id = d.dept_id
        JOIN department cd ON c.dept_id = cd.dept_id
        LEFT JOIN course_difficulty cd_diff ON c.course_id = cd_diff.course_id
        LEFT JOIN (
            SELECT stu_id, AVG(attendance_percentage) as avg_attendance
            FROM attendance a
            JOIN enrollment e ON a.enroll_id = e.enroll_id
            GROUP BY stu_id
        ) avg_attendance ON s.stu_id = avg_attendance.stu_id
        LEFT JOIN (
            SELECT stu_id, AVG(grade) as avg_grade
            FROM enrollment
            GROUP BY stu_id
        ) avg_grade ON s.stu_id = avg_grade.stu_id
        WHERE NOT EXISTS (
            SELECT 1 FROM enrollment e 
            WHERE e.stu_id = s.stu_id AND e.course_id = c.course_id AND e.semester_id = sem.semester_id
        )
        LIMIT 1000
        """
        
        with connection.cursor() as cursor:
            cursor.execute(query)
            positive_data = cursor.fetchall()
            
            cursor.execute(negative_query)
            negative_data = cursor.fetchall()
        
        columns = ['stu_id', 'course_id', 'semester_id', 'gender', 'student_dept', 
                    'course_dept', 'difficulty_level', 'historical_attendance', 
                    'historical_grade', 'enrolled']
        
        df_positive = pd.DataFrame(positive_data, columns=columns)
        df_negative = pd.DataFrame(negative_data, columns=columns)
        
        df = pd.concat([df_positive, df_negative], ignore_index=True)
        
        return df

    def prepare_features(self, df):
        """Prepare features for training"""
        # Encode categorical variables
        le_gender = LabelEncoder()
        le_student_dept = LabelEncoder()
        le_course_dept = LabelEncoder()
        le_difficulty = LabelEncoder()
        
        df_encoded = df.copy()
        df_encoded['gender_encoded'] = le_gender.fit_transform(df['gender'])
        df_encoded['student_dept_encoded'] = le_student_dept.fit_transform(df['student_dept'])
        df_encoded['course_dept_encoded'] = le_course_dept.fit_transform(df['course_dept'])
        df_encoded['difficulty_encoded'] = le_difficulty.fit_transform(df['difficulty_level'])
        
        # Same department indicator
        df_encoded['same_dept'] = (df['student_dept'] == df['course_dept']).astype(int)
        
        # Feature columns
        feature_columns = ['gender_encoded', 'student_dept_encoded', 'course_dept_encoded', 
                            'difficulty_encoded', 'historical_attendance', 'historical_grade', 'same_dept']
        
        X = df_encoded[feature_columns]
        y = df_encoded['enrolled']
        
        # Save encoders
        encoders = {
            'gender': le_gender,
            'student_dept': le_student_dept,
            'course_dept': le_course_dept,
            'difficulty': le_difficulty
        }
        
        joblib.dump(encoders, 'course_enrollment_encoders.pkl')
        
        return X, y, feature_columns

    def train_model(self):
        """Train the enrollment prediction model"""
        self.stdout.write("Loading data...")
        df = self.get_data()
        
        self.stdout.write("Preparing features...")
        X, y, feature_columns = self.prepare_features(df)
        
        self.stdout.write("Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        self.stdout.write("Training model...")
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Save model
        joblib.dump(model, 'course_enrollment_model.pkl')
        joblib.dump(feature_columns, 'course_enrollment_features.pkl')
        
        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        self.stdout.write(f"Model trained successfully!")
        self.stdout.write(f"Accuracy: {accuracy:.4f}")
        self.stdout.write(f"Classification Report:")
        self.stdout.write(classification_report(y_test, y_pred))
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        self.stdout.write("Feature Importance:")
        self.stdout.write(str(feature_importance))

    def test_model(self):
        """Test the trained model"""
        # Load model and encoders
        model = joblib.load('course_enrollment_model.pkl')
        encoders = joblib.load('course_enrollment_encoders.pkl')
        feature_columns = joblib.load('course_enrollment_features.pkl')
        
        # Get test data
        df = self.get_data()
        X, y, _ = self.prepare_features(df)
        
        # Predictions
        y_pred = model.predict(X)
        y_pred_proba = model.predict_proba(X)
        
        # Metrics
        accuracy = accuracy_score(y, y_pred)
        self.stdout.write(f"Test Accuracy: {accuracy:.4f}")
        self.stdout.write(f"Classification Report:")
        self.stdout.write(classification_report(y, y_pred))
        
        # Generate visualizations
        self.generate_visualizations(y, y_pred, y_pred_proba, model, feature_columns)

    def predict_enrollment(self, student_id, course_id, semester_id):
        """Predict enrollment for specific student-course-semester combination"""
        if not all([student_id, course_id, semester_id]):
            self.stdout.write("Please provide student_id, course_id, and semester_id")
            return
        
        # Load model and encoders
        model = joblib.load('course_enrollment_model.pkl')
        encoders = joblib.load('course_enrollment_encoders.pkl')
        
        # Get student and course information
        query = """
        SELECT 
            s.gender,
            d.dept_name as student_dept,
            cd.dept_name as course_dept,
            CASE WHEN cd_diff.difficulty_level IS NOT NULL THEN cd_diff.difficulty_level ELSE 'Medium' END as difficulty_level,
            COALESCE(avg_attendance.avg_attendance, 75) as historical_attendance,
            COALESCE(avg_grade.avg_grade, 75) as historical_grade
        FROM student s
        JOIN course c ON c.course_id = %s
        JOIN department d ON s.dept_id = d.dept_id
        JOIN department cd ON c.dept_id = cd.dept_id
        LEFT JOIN course_difficulty cd_diff ON c.course_id = cd_diff.course_id
        LEFT JOIN (
            SELECT stu_id, AVG(attendance_percentage) as avg_attendance
            FROM attendance a
            JOIN enrollment e ON a.enroll_id = e.enroll_id
            WHERE stu_id = %s
            GROUP BY stu_id
        ) avg_attendance ON s.stu_id = avg_attendance.stu_id
        LEFT JOIN (
            SELECT stu_id, AVG(grade) as avg_grade
            FROM enrollment
            WHERE stu_id = %s
            GROUP BY stu_id
        ) avg_grade ON s.stu_id = avg_grade.stu_id
        WHERE s.stu_id = %s
        """
        
        with connection.cursor() as cursor:
            cursor.execute(query, [course_id, student_id, student_id, student_id])
            result = cursor.fetchone()
        
        if not result:
            self.stdout.write("Student or course not found")
            return
        
        # Prepare features
        gender_encoded = encoders['gender'].transform([result[0]])[0]
        student_dept_encoded = encoders['student_dept'].transform([result[1]])[0]
        course_dept_encoded = encoders['course_dept'].transform([result[2]])[0]
        difficulty_encoded = encoders['difficulty'].transform([result[3]])[0]
        same_dept = 1 if result[1] == result[2] else 0
        
        features = np.array([[gender_encoded, student_dept_encoded, course_dept_encoded, 
                            difficulty_encoded, result[4], result[5], same_dept]])
        
        # Predict
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0]
        
        self.stdout.write(f"Enrollment Prediction for Student {student_id}, Course {course_id}, Semester {semester_id}:")
        self.stdout.write(f"Will Enroll: {'Yes' if prediction == 1 else 'No'}")
        self.stdout.write(f"Probability of Enrollment: {probability[1]:.4f}")
        self.stdout.write(f"Probability of Not Enrolling: {probability[0]:.4f}")

    def generate_visualizations(self, y_true, y_pred, y_pred_proba, model, feature_columns):
        """Generate visualization plots"""
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0,0])
        axes[0,0].set_title('Confusion Matrix')
        axes[0,0].set_xlabel('Predicted')
        axes[0,0].set_ylabel('Actual')
        
        # Feature Importance
        feature_importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=True)
        
        axes[0,1].barh(feature_importance['feature'], feature_importance['importance'])
        axes[0,1].set_title('Feature Importance')
        axes[0,1].set_xlabel('Importance')
        
        # Prediction Probability Distribution
        axes[1,0].hist(y_pred_proba[:, 1], bins=50, alpha=0.7, color='skyblue')
        axes[1,0].set_title('Enrollment Probability Distribution')
        axes[1,0].set_xlabel('Probability of Enrollment')
        axes[1,0].set_ylabel('Frequency')
        
        # ROC Curve (simplified)
        from sklearn.metrics import roc_curve, auc
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        
        axes[1,1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        axes[1,1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axes[1,1].set_xlim([0.0, 1.0])
        axes[1,1].set_ylim([0.0, 1.05])
        axes[1,1].set_xlabel('False Positive Rate')
        axes[1,1].set_ylabel('True Positive Rate')
        axes[1,1].set_title('ROC Curve')
        axes[1,1].legend(loc="lower right")
        
        plt.tight_layout()
        plt.savefig('course_enrollment_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        self.stdout.write("Visualizations saved as 'course_enrollment_analysis.png'")