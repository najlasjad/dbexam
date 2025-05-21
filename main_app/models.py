from django.db import models

# Create your models here.
from django.db import models

class Department(models.Model):
    dept_id = models.IntegerField(primary_key=True)
    dept_name = models.CharField(max_length=100)

    def __str__(self):
        return self.dept_name
    
    class Meta:
        db_table = 'department'
        managed = False

class Student(models.Model):
    stu_id = models.IntegerField(primary_key=True)
    name = models.CharField(max_length=100)
    email = models.CharField(max_length=100)
    gender = models.CharField(max_length=10)
    dob = models.DateField()
    dept_id = models.ForeignKey(Department, on_delete=models.CASCADE, db_column='dept_id')

    def __str__(self):
        return self.name
    
    class Meta:
        db_table = 'student'
        managed = False

class Course(models.Model):
    course_id = models.IntegerField(primary_key=True)
    course_name = models.CharField(max_length=100)
    dept_id = models.ForeignKey(Department, on_delete=models.CASCADE, db_column='dept_id')

    def __str__(self):
        return self.course_name
    
    class Meta:
        db_table = 'course'
        managed = False

class Semester(models.Model):
    semester_id = models.IntegerField(primary_key=True)
    semester_name = models.CharField(max_length=100)

    def __str__(self):
        return self.semester_name
    
    class Meta:
        db_table = 'semester'
        managed = False

class Enrollment(models.Model):
    enroll_id = models.IntegerField(primary_key=True)
    stu_id = models.ForeignKey(Student, on_delete=models.CASCADE, db_column='stu_id')
    course_id = models.ForeignKey(Course, on_delete=models.CASCADE, db_column='course_id')
    semester_id = models.ForeignKey(Semester, on_delete=models.CASCADE, db_column='semester_id')
    grade = models.IntegerField()

    class Meta:
        db_table = 'enrollment'
        managed = False
    

class Assessment(models.Model):
    assessment_id = models.IntegerField(primary_key=True)
    enroll_id = models.ForeignKey(Enrollment, on_delete=models.CASCADE, db_column='enroll_id')
    assessment_type = models.CharField(max_length=50)
    score = models.IntegerField()

    class Meta:
        db_table = 'assessment'
        managed  = False

class Attendance(models.Model):
    attendance_id = models.IntegerField(primary_key=True)
    enroll_id = models.ForeignKey(Enrollment, on_delete=models.CASCADE, db_column='enroll_id')
    attandance_percentage = models.IntegerField()

    class Meta:
        db_table = 'attendance'
        managed = False

class CourseDifficulty(models.Model):
    course_id = models.ForeignKey(Course, on_delete=models.CASCADE, db_column='course_id')
    difficulty_level = models.CharField(max_length=50)

    class Meta:
        db_table = 'course_difficulty'
        managed = False

class Instructor(models.Model):
    instructor_id = models.IntegerField(primary_key=True)
    instructor_name = models.CharField(max_length=100)
    dept_id = models.ForeignKey(Department, on_delete=models.CASCADE, db_column='dept_id')

    def __str__(self):
        return self.instructor_name

    class Meta:
        db_table = 'instructor'
        managed = False

class CourseInstructor(models.Model):
    course_instructor_id = models.IntegerField(primary_key=True)
    course_id = models.ForeignKey(Course, on_delete=models.CASCADE, db_column='course_id')
    instructor_id = models.ForeignKey(Instructor, on_delete=models.CASCADE, db_column='instructor_id')
    semester_id = models.ForeignKey(Semester, on_delete=models.CASCADE, db_column='semester_id')

    class Meta:
        db_table = 'course_instructor'
        managed = False

class CourseSemester(models.Model):
    id = models.IntegerField(primary_key=True)
    course_id = models.ForeignKey(Course, on_delete=models.CASCADE, db_column='course_id')
    semester_id = models.ForeignKey(Semester, on_delete=models.CASCADE, db_column='semester_id')

    class Meta:
        db_table = 'course_semester'
        managed = False