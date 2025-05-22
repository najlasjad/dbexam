from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
from django.contrib.auth.decorators import login_required
from django.db import connection
from django.views.decorators.csrf import csrf_exempt
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO
import os
from django.db.models import Case, When

@login_required
def dashboard_home(request):
    """Main dashboard home page"""
    return render(request, 'najla_app/dashboard.html')

@login_required
def enrollment_prediction_dashboard(request):
    """Course Enrollment Prediction Dashboard"""
    context = {
        'page_title': 'Course Enrollment Prediction',
        'description': 'Predict student enrollment likelihood for courses'
    }
    
    if request.method == 'GET':
        # Get students and courses for dropdown
        with connection.cursor() as cursor:
            cursor.execute("SELECT stu_id, name FROM student ORDER BY name")
            students = cursor.fetchall()
            
            cursor.execute("SELECT course_id, course_name FROM course ORDER BY course_name")
            courses = cursor.fetchall()
            
            cursor.execute("SELECT semester_id, semester_name FROM semester ORDER BY semester_id")
            semesters = cursor.fetchall()
        
        context.update({
            'students': students,
            'courses': courses,
            'semesters': semesters
        })
        
    return render(request, 'najla_app/enrollment_prediction.html', context)

@csrf_exempt
@login_required
def predict_enrollment(request):
    """API endpoint for enrollment prediction"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            student_id = data.get('student_id')
            course_id = data.get('course_id')
            semester_id = data.get('semester_id')
            
            # Load model and make prediction
            try:
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
                
                if result:
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
                    
                    return JsonResponse({
                        'success': True,
                        'prediction': int(prediction),
                        'probability_enroll': float(probability[1]),
                        'probability_not_enroll': float(probability[0]),
                        'recommendation': 'Likely to Enroll' if prediction == 1 else 'Unlikely to Enroll'
                    })
                else:
                    return JsonResponse({'success': False, 'error': 'Student or course not found'})
                    
            except FileNotFoundError:
                return JsonResponse({'success': False, 'error': 'Model not trained yet'})
                
        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)})
    
    return JsonResponse({'success': False, 'error': 'Invalid request method'})

@login_required
def grade_trend_dashboard(request):
    """Grade Trend Analysis Dashboard"""
    context = {
        'page_title': 'Grade Trend Analysis',
        'description': 'Analyze and predict grade trends across semesters'
    }
    
    # Get trend data for visualization
    with connection.cursor() as cursor:
        # Semester trends
        cursor.execute("""
            SELECT sem.semester_name, AVG(e.grade) as avg_grade, COUNT(*) as student_count
            FROM enrollment e
            JOIN semester sem ON e.semester_id = sem.semester_id
            GROUP BY sem.semester_id, sem.semester_name
            ORDER BY sem.semester_id
        """)
        semester_trends = cursor.fetchall()
        
        # Department trends
        cursor.execute("""
            SELECT d.dept_name, AVG(e.grade) as avg_grade, COUNT(*) as student_count
            FROM enrollment e
            JOIN student s ON e.stu_id = s.stu_id
            JOIN department d ON s.dept_id = d.dept_id
            GROUP BY d.dept_id, d.dept_name
            ORDER BY avg_grade DESC
        """)
        dept_trends = cursor.fetchall()
        
        # Course difficulty impact
        cursor.execute("""
            SELECT 
                COALESCE(cd.difficulty_level, 'Medium') as difficulty,
                AVG(e.grade) as avg_grade,
                COUNT(*) as course_count
            FROM enrollment e
            JOIN course c ON e.course_id = c.course_id
            LEFT JOIN course_difficulty cd ON c.course_id = cd.course_id
            GROUP BY difficulty
        """)
        difficulty_trends = cursor.fetchall()
    
    context.update({
        'semester_trends': semester_trends,
        'dept_trends': dept_trends,
        'difficulty_trends': difficulty_trends
    })
    
    return render(request, 'najla_app/grade_trend.html', context)

@csrf_exempt
@login_required
def predict_future_grades(request):
    """API endpoint for future grade prediction"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            student_id = data.get('student_id')
            future_semesters = data.get('future_semesters', 2)
            
            try:
                model = joblib.load('grade_trend_model.pkl')
                scaler = joblib.load('grade_trend_scaler.pkl')
                
                # Get student's historical data
                query = """
                SELECT 
                    e.semester_id,
                    e.grade,
                    COALESCE(cd.difficulty_level, 'Medium') as difficulty_level,
                    COALESCE(att.attendance_percentage, 75) as attendance,
                    COALESCE(avg_assess.avg_assessment, 75) as avg_assessment_score
                FROM enrollment e
                JOIN course c ON e.course_id = c.course_id
                LEFT JOIN course_difficulty cd ON c.course_id = cd.course_id
                LEFT JOIN attendance att ON e.enroll_id = att.enroll_id
                LEFT JOIN (
                    SELECT enroll_id, AVG(score) as avg_assessment
                    FROM assessment
                    GROUP BY enroll_id
                ) avg_assess ON e.enroll_id = avg_assess.enroll_id
                WHERE e.stu_id = %s
                ORDER BY e.semester_id DESC
                LIMIT 5
                """
                
                with connection.cursor() as cursor:
                    cursor.execute(query, [student_id])
                    historical_data = cursor.fetchall()
                
                if historical_data:
                    latest_semester = historical_data[0][0]
                    latest_grade = historical_data[0][1]
                    avg_grade = sum([row[1] for row in historical_data]) / len(historical_data)
                    
                    predictions = []
                    current_grade = latest_grade
                    
                    difficulty_map = {'Easy': 1, 'Medium': 2, 'Hard': 3}
                    avg_difficulty = np.mean([difficulty_map.get(row[2], 2) for row in historical_data])
                    avg_attendance = np.mean([row[3] for row in historical_data])
                    avg_assessment = np.mean([row[4] for row in historical_data])
                    
                    for i in range(1, future_semesters + 1):
                        # Simple prediction based on trend
                        feature_values = [
                            latest_semester + i,  # semester_order
                            current_grade,        # prev_grade
                            avg_grade,           # prev_semester_avg
                            avg_difficulty,      # difficulty_numeric
                            avg_attendance,      # attendance
                            avg_assessment,      # avg_assessment_score
                            avg_grade,           # historical_grade_mean
                            np.std([row[1] for row in historical_data]),  # historical_grade_std
                            len(historical_data)  # course_count
                        ]
                        
                        # Scale and predict
                        X_pred = scaler.transform([feature_values])
                        predicted_grade = model.predict(X_pred)[0]
                        
                        predictions.append({
                            'semester': latest_semester + i,
                            'predicted_grade': round(predicted_grade, 2),
                            'grade_change': round(predicted_grade - current_grade, 2)
                        })
                        
                        current_grade = predicted_grade
                    
                    return JsonResponse({
                        'success': True,
                        'current_grade': latest_grade,
                        'current_semester': latest_semester,
                        'predictions': predictions
                    })
                else:
                    return JsonResponse({'success': False, 'error': 'No historical data found'})
                    
            except FileNotFoundError:
                return JsonResponse({'success': False, 'error': 'Model not trained yet'})
                
        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)})
    
    return JsonResponse({'success': False, 'error': 'Invalid request method'})

@login_required
def instructor_effectiveness_dashboard(request):
    """Instructor Effectiveness Dashboard"""
    context = {
        'page_title': 'Instructor Effectiveness Analysis',
        'description': 'Analyze instructor impact on student performance'
    }
    
    # Get instructor effectiveness data
    with connection.cursor() as cursor:
        # Top performing instructors
        cursor.execute("""
            SELECT 
                i.instructor_id,
                i.instructor_name,
                AVG(e.grade) as avg_grade,
                COUNT(DISTINCT e.stu_id) as total_students,
                COUNT(DISTINCT CONCAT(ci.course_id, '-', ci.semester_id)) as classes_taught,
                STDDEV(e.grade) as grade_consistency
            FROM instructor i
            JOIN course_instructor ci ON i.instructor_id = ci.instructor_id
            JOIN enrollment e ON (e.course_id = ci.course_id AND e.semester_id = ci.semester_id)
            GROUP BY i.instructor_id, i.instructor_name
            HAVING COUNT(DISTINCT e.stu_id) >= 5
            ORDER BY avg_grade DESC
            LIMIT 10
        """)
        top_instructors = cursor.fetchall()
        
        # Course baseline vs instructor performance
        cursor.execute("""
            WITH course_baselines AS (
                SELECT course_id, AVG(grade) as baseline_grade
                FROM enrollment
                GROUP BY course_id
            ),
            instructor_performance AS (
                SELECT 
                    ci.instructor_id,
                    i.instructor_name,
                    ci.course_id,
                    c.course_name,
                    AVG(e.grade) as instructor_avg_grade,
                    cb.baseline_grade,
                    AVG(e.grade) - cb.baseline_grade as effectiveness
                FROM course_instructor ci
                JOIN instructor i ON ci.instructor_id = i.instructor_id
                JOIN course c ON ci.course_id = c.course_id
                JOIN enrollment e ON (e.course_id = ci.course_id AND e.semester_id = ci.semester_id)
                JOIN course_baselines cb ON ci.course_id = cb.course_id
                GROUP BY ci.instructor_id, i.instructor_name, ci.course_id, c.course_name, cb.baseline_grade
            )
            SELECT 
                instructor_name,
                AVG(effectiveness) as avg_effectiveness,
                COUNT(*) as courses_taught
            FROM instructor_performance
            GROUP BY instructor_id, instructor_name
            ORDER BY avg_effectiveness DESC
            LIMIT 15
        """)
        effectiveness_rankings = cursor.fetchall()
        
        # Department wise instructor performance
        cursor.execute("""
            SELECT 
                d.dept_name,
                COUNT(DISTINCT i.instructor_id) as instructor_count,
                AVG(e.grade) as dept_avg_grade,
                STDDEV(e.grade) as grade_variance
            FROM instructor i
            JOIN department d ON i.dept_id = d.dept_id
            JOIN course_instructor ci ON i.instructor_id = ci.instructor_id
            JOIN enrollment e ON (e.course_id = ci.course_id AND e.semester_id = ci.semester_id)
            GROUP BY d.dept_id, d.dept_name
            ORDER BY dept_avg_grade DESC
        """)
        dept_performance = cursor.fetchall()
    
    context.update({
        'top_instructors': top_instructors,
        'effectiveness_rankings': effectiveness_rankings,
        'dept_performance': dept_performance
    })
    
    return render(request, 'najla_app/instructor_effectiveness.html', context)

@csrf_exempt
@login_required
def analyze_instructor_impact(request):
    """API endpoint for instructor impact analysis"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            instructor_id = data.get('instructor_id')
            
            # Get instructor analysis
            with connection.cursor() as cursor:
                # Instructor overview
                cursor.execute("""
                    SELECT 
                        i.instructor_name,
                        d.dept_name,
                        AVG(e.grade) as avg_grade,
                        STDDEV(e.grade) as grade_consistency,
                        COUNT(DISTINCT e.stu_id) as total_students,
                        COUNT(DISTINCT CONCAT(ci.course_id, '-', ci.semester_id)) as classes_taught,
                        AVG(COALESCE(att.attendance_percentage, 75)) as avg_class_attendance
                    FROM instructor i
                    JOIN department d ON i.dept_id = d.dept_id
                    JOIN course_instructor ci ON i.instructor_id = ci.instructor_id
                    JOIN enrollment e ON (e.course_id = ci.course_id AND e.semester_id = ci.semester_id)
                    LEFT JOIN attendance att ON e.enroll_id = att.enroll_id
                    WHERE i.instructor_id = %s
                    GROUP BY i.instructor_id, i.instructor_name, d.dept_name
                """, [instructor_id])
                instructor_overview = cursor.fetchone()
                
                # Course-wise performance
                cursor.execute("""
                    SELECT 
                        c.course_name,
                        sem.semester_name,
                        AVG(e.grade) as avg_grade,
                        COUNT(e.stu_id) as student_count
                    FROM course_instructor ci
                    JOIN course c ON ci.course_id = c.course_id
                    JOIN semester sem ON ci.semester_id = sem.semester_id
                    JOIN enrollment e ON (e.course_id = ci.course_id AND e.semester_id = ci.semester_id)
                    WHERE ci.instructor_id = %s
                    GROUP BY c.course_id, c.course_name, sem.semester_id, sem.semester_name
                    ORDER BY sem.semester_id DESC, c.course_name
                """, [instructor_id])
                course_performance = cursor.fetchall()
                
                # Effectiveness compared to course baseline
                cursor.execute("""
                    WITH course_baselines AS (
                        SELECT course_id, AVG(grade) as baseline_grade
                        FROM enrollment
                        GROUP BY course_id
                    )
                    SELECT 
                        c.course_name,
                        AVG(e.grade) as instructor_avg,
                        cb.baseline_grade,
                        AVG(e.grade) - cb.baseline_grade as effectiveness
                    FROM course_instructor ci
                    JOIN course c ON ci.course_id = c.course_id
                    JOIN enrollment e ON (e.course_id = ci.course_id AND e.semester_id = ci.semester_id)
                    JOIN course_baselines cb ON ci.course_id = cb.course_id
                    WHERE ci.instructor_id = %s
                    GROUP BY c.course_id, c.course_name, cb.baseline_grade
                    ORDER BY effectiveness DESC
                """, [instructor_id])
                effectiveness_data = cursor.fetchall()
            
            if instructor_overview:
                return JsonResponse({
                    'success': True,
                    'instructor_overview': {
                        'name': instructor_overview[0],
                        'department': instructor_overview[1],
                        'avg_grade': round(instructor_overview[2], 2),
                        'grade_consistency': round(instructor_overview[3], 2) if instructor_overview[3] else 0,
                        'total_students': instructor_overview[4],
                        'classes_taught': instructor_overview[5],
                        'avg_attendance': round(instructor_overview[6], 2)
                    },
                    'course_performance': [
                        {
                            'course': row[0],
                            'semester': row[1],
                            'avg_grade': round(row[2], 2),
                            'student_count': row[3]
                        }
                        for row in course_performance
                    ],
                    'effectiveness_analysis': [
                        {
                            'course': row[0],
                            'instructor_avg': round(row[1], 2),
                            'baseline': round(row[2], 2),
                            'effectiveness': round(row[3], 2)
                        }
                        for row in effectiveness_data
                    ]
                })
            else:
                return JsonResponse({'success': False, 'error': 'Instructor not found'})
                
        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)})
    
    return JsonResponse({'success': False, 'error': 'Invalid request method'})

@login_required
def generate_chart(request):
    """Generate charts for dashboard"""
    chart_type = request.GET.get('type')
    
    try:
        plt.style.use('default')
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if chart_type == 'semester_trends':
            with connection.cursor() as cursor:
                cursor.execute("""
                    SELECT sem.semester_name, AVG(e.grade) as avg_grade
                    FROM enrollment e
                    JOIN semester sem ON e.semester_id = sem.semester_id
                    GROUP BY sem.semester_id, sem.semester_name
                    ORDER BY sem.semester_id
                """)
                data = cursor.fetchall()
            
            semesters = [row[0] for row in data]
            grades = [row[1] for row in data]
            
            ax.plot(semesters, grades, marker='o', linewidth=2, markersize=8)
            ax.set_title('Grade Trends Across Semesters', fontsize=16, fontweight='bold')
            ax.set_xlabel('Semester', fontsize=12)
            ax.set_ylabel('Average Grade', fontsize=12)
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            
        elif chart_type == 'dept_performance':
            with connection.cursor() as cursor:
                cursor.execute("""
                    SELECT d.dept_name, AVG(e.grade) as avg_grade
                    FROM enrollment e
                    JOIN student s ON e.stu_id = s.stu_id
                    JOIN department d ON s.dept_id = d.dept_id
                    GROUP BY d.dept_id, d.dept_name
                    ORDER BY avg_grade DESC
                """)
                data = cursor.fetchall()
            
            depts = [row[0] for row in data]
            grades = [row[1] for row in data]
            
            bars = ax.bar(depts, grades, color='skyblue', edgecolor='navy', alpha=0.7)
            ax.set_title('Department Performance Comparison', fontsize=16, fontweight='bold')
            ax.set_ylabel('Average Grade', fontsize=12)
            plt.xticks(rotation=45)
            
            # Add value labels on bars
            for bar, grade in zip(bars, grades):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{grade:.1f}', ha='center', va='bottom', fontweight='bold')
                
        elif chart_type == 'instructor_effectiveness':
            with connection.cursor() as cursor:
                cursor.execute("""
                    WITH course_baselines AS (
                        SELECT course_id, AVG(grade) as baseline_grade
                        FROM enrollment
                        GROUP BY course_id
                    ),
                    instructor_effectiveness AS (
                        SELECT 
                            i.instructor_name,
                            AVG(e.grade) - AVG(cb.baseline_grade) as effectiveness
                        FROM instructor i
                        JOIN course_instructor ci ON i.instructor_id = ci.instructor_id
                        JOIN enrollment e ON (e.course_id = ci.course_id AND e.semester_id = ci.semester_id)
                        JOIN course_baselines cb ON ci.course_id = cb.course_id
                        GROUP BY i.instructor_id, i.instructor_name
                        HAVING COUNT(DISTINCT e.stu_id) >= 10
                    )
                    SELECT instructor_name, effectiveness
                    FROM instructor_effectiveness
                    ORDER BY effectiveness DESC
                    LIMIT 15
                """)
                data = cursor.fetchall()
            
            instructors = [row[0].split()[-1] for row in data]  # Last name only
            effectiveness = [row[1] for row in data]
            
            colors = ['green' if x > 0 else 'red' for x in effectiveness]
            bars = ax.barh(instructors, effectiveness, color=colors, alpha=0.7)
            ax.set_title('Instructor Effectiveness (vs Course Baseline)', fontsize=16, fontweight='bold')
            ax.set_xlabel('Effectiveness Score', fontsize=12)
            ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            
        # Convert plot to base64 string
        buffer = BytesIO()
        plt.tight_layout()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        plt.close()
        
        graphic = base64.b64encode(image_png)
        graphic = graphic.decode('utf-8')
        
        return JsonResponse({
            'success': True,
            'chart': graphic
        })
        
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        })

@login_required
def get_dashboard_stats(request):
    """Get overall dashboard statistics"""
    try:
        with connection.cursor() as cursor:
            # Overall stats
            cursor.execute("SELECT COUNT(*) FROM student")
            total_students = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM instructor")
            total_instructors = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM course")
            total_courses = cursor.fetchone()[0]
            
            cursor.execute("SELECT AVG(grade) FROM enrollment")
            overall_avg_grade = cursor.fetchone()[0]
            
            cursor.execute("SELECT AVG(attendance_percentage) FROM attendance")
            overall_avg_attendance = cursor.fetchone()[0]
            
            # Recent trends
            cursor.execute("""
                SELECT 
                    COUNT(*) as enrollments,
                    AVG(grade) as avg_grade
                FROM enrollment e
                JOIN semester s ON e.semester_id = s.semester_id
                WHERE e.semester_id = (SELECT MAX(semester_id) FROM semester)
            """)
            recent_stats = cursor.fetchone()
            
        return JsonResponse({
            'success': True,
            'stats': {
                'total_students': total_students,
                'total_instructors': total_instructors,
                'total_courses': total_courses,
                'overall_avg_grade': round(overall_avg_grade, 2) if overall_avg_grade else 0,
                'overall_avg_attendance': round(overall_avg_attendance, 2) if overall_avg_attendance else 0,
                'recent_enrollments': recent_stats[0] if recent_stats else 0,
                'recent_avg_grade': round(recent_stats[1], 2) if recent_stats and recent_stats[1] else 0
            }
        })
        
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        })