import logging
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login as auth_login, logout as auth_logout
from django.contrib.auth.decorators import login_required, user_passes_test
from .forms import CustomUserCreationForm
from najla_app.views import dashboard_home

# Buat logger
logger = logging.getLogger(__name__)

# SIGNUP VIEW
def signup(request):
    if request.method == 'POST':
        form = CustomUserCreationForm(request.POST)
        if form.is_valid():
            form.save()
            logger.info('User signed up successfully')
            return redirect('login')
        else:
            logger.error('Signup failed. Form is invalid.')
    else:
        form = CustomUserCreationForm()
    return render(request, 'registration/signup.html', {'form': form})

# LOGIN VIEW
def login(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(request, username=username, password=password)
        if user is not None:
            auth_login(request, user)
            logger.info('User logged in successfully')
            return redirect('home')
        else:
            logger.warning('Invalid login attempt')
            return render(request, 'registration/login.html', {'error': 'Invalid credentials'})
    return render(request, 'registration/login.html')

# LOGOUT VIEW
def logout(request):
    auth_logout(request)
    logger.info('User logged out successfully')
    return redirect('login')

# HOME VIEW
@login_required
def home(request):
    logger.info(f'User {request.user.username} accessed home page')
    return render(request, 'home.html')

# ABOUT VIEW
@login_required
def about(request):
    logger.info(f'User {request.user.username} accessed about page')
    return render(request, 'about.html')

# USECASE VIEW
@login_required
def usecase(request):
    return dashboard_home(request)

# MODEL VIEW (hanya superuser yang boleh)
@login_required
@user_passes_test(lambda u: u.is_superuser)
def model(request):
    logger.info(f'Superuser {request.user.username} accessed model page')
    return render(request, 'model.html')
