from django.db import models

# Create your models here.
from django.db import models

class Department(models.Model):
    dept_name = models.CharField(max_length=100)

    def __str__(self):
        return self.dept_name
    
    class Meta:
        app_label = 'main_app'

class Student(models.Model):
    name = models.CharField(max_length=100)
    email = models.CharField(max_length=100)
    gender = models.CharField(max_length=10)
    dob = models.DateField()
    dept = models.ForeignKey(Department, on_delete=models.CASCADE)

    def __str__(self):
        return self.name

class Instructor(models.Model):
    instructor_name = models.CharField(max_length=100)
    dept = models.ForeignKey(Department, on_delete=models.CASCADE)

    def __str__(self):
        return self.instructor_name

class Course(models.Model):
    course_name = models.CharField(max_length=100)
    dept = models.ForeignKey(Department, on_delete=models.CASCADE)

    def __str__(self):
        return self.course_name

class Semester(models.Model):
    semester_name = models.CharField(max_length=100)

    def __str__(self):
        return self.semester_name

class CourseDifficulty(models.Model):
    course = models.OneToOneField(Course, on_delete=models.CASCADE)
    difficulty_level = models.CharField(max_length=50)

class CourseInstructor(models.Model):
    course = models.ForeignKey(Course, on_delete=models.CASCADE)
    instructor = models.ForeignKey(Instructor, on_delete=models.CASCADE)
    semester = models.ForeignKey(Semester, on_delete=models.CASCADE)

class CourseSemester(models.Model):
    course = models.ForeignKey(Course, on_delete=models.CASCADE)
    semester = models.ForeignKey(Semester, on_delete=models.CASCADE)

class Enrollment(models.Model):
    stu = models.ForeignKey(Student, on_delete=models.CASCADE)
    course = models.ForeignKey(Course, on_delete=models.CASCADE)
    semester = models.ForeignKey(Semester, on_delete=models.CASCADE)
    grade = models.IntegerField()

class Assessment(models.Model):
    enroll = models.ForeignKey(Enrollment, on_delete=models.CASCADE)
    assessment_type = models.CharField(max_length=50)
    score = models.IntegerField()

class Attendance(models.Model):
    enroll = models.ForeignKey(Enrollment, on_delete=models.CASCADE)
    attendance_percentage = models.IntegerField()
