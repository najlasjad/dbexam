from django.shortcuts import render

def home(request):
    return render(request, 'home.html')

def about(request):
    return render(request, 'about.html')

def usecase(request):
    return render(request, 'usecase.html')

def model(request):
    return render(request, 'model.html')