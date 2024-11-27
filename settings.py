import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from the .env file
load_dotenv("C:\\bot\\.env")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
NASDAQ_DATA_LINK_API_KEY = os.getenv("NASDAQ_DATA_LINK_API_KEY")

# Build paths inside the project like this: os.path.join(BASE_DIR, ...)
BASE_DIR = Path(__file__).resolve().parent.parent

# Quick-start development settings - unsuitable for production
SECRET_KEY = os.getenv('DJANGO_SECRET_KEY', 'your-default-secret-key')
DEBUG = True  # Change to False for production

ALLOWED_HOSTS = [
    '127.0.0.1',
    'localhost',
    '7581-2600-4040-904c-4e00-d5e5-f0db-f77e-b9d5.ngrok-free.app'
]

CSRF_TRUSTED_ORIGINS = [
    'https://7581-2600-4040-904c-4e00-d5e5-f0db-f77e-b9d5.ngrok-free.app'
]

# Application definition
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'chatbot_app',  # Replace with your app name
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'bot.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [
            os.path.join(BASE_DIR, 'templates'),  # Look for global templates
            os.path.join(BASE_DIR, 'chatbot_app', 'templates'),  # Look in the app-specific templates
        ],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'bot.wsgi.application'

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': r'C:\sqlite-tools\datanew.db',  # Ensure the file exists here
    }
}

LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
        },
    },
    'root': {
        'handlers': ['console'],
        'level': 'DEBUG',
    },
}


# Internationalization
LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_L10N = True
USE_TZ = True

# Static files (CSS, JavaScript, Images)
STATIC_URL = '/static/'

STATICFILES_DIRS = [
    r'C:\bot\chatbot_app\static',  # Main static directory
]

STATIC_ROOT = r'C:\bot\staticfiles'  # Directory for `collectstatic` output

# settings.py or views.py
import os
from dotenv import load_dotenv

load_dotenv()  # Ensure the .env file is loaded
NASDAQ_API_KEY = os.getenv("NASDAQ_DATA_LINK_API_KEY")

