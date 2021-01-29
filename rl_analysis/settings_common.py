"""
Django settings for rl_analysis project.

Generated by 'django-admin startproject' using Django 2.2.2.

For more information on this file, see
https://docs.djangoproject.com/en/2.2/topics/settings/

For the full list of settings and their values, see
https://docs.djangoproject.com/en/2.2/ref/settings/
"""

import os



# Build paths inside the project like this: os.path.join(BASE_DIR, ...)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/2.2/howto/deployment/checklist/

# Application definition

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'reversi.apps.ReversiConfig',
    'accounts.apps.AccountsConfig',

    # django-allauth
    'django.contrib.sites',
    'allauth',
    'allauth.account',
    'allauth.socialaccount'
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
    'whitenoise.middleware.WhiteNoiseMiddleware',
    'accounts.middleware.auth.authMiddleware',
]

ROOT_URLCONF = 'rl_analysis.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
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

WSGI_APPLICATION = 'rl_analysis.wsgi.application'


# Database
# https://docs.djangoproject.com/en/2.2/ref/settings/#databases



AUTH_USER_MODEL = 'accounts.CustomUser'

# django.contrib.sitesを使うためのサイト識別用ID
SITE_ID = 1

# 認証バックエンド(認証をテストするクラス)を設定
AUTHENTICATION_BACKENDS = (
    'allauth.account.auth_backends.AuthenticationBackend',
    # 一般ユーザー用
    'django.contrib.auth.backends.ModelBackend',
    # 管理サイト用
)

# メールアドレス認証かユーザー名でログイン
ACCOUNT_AUTHENICATION_METHOD = 'username'
ACCOUNT_USERNAME_REQUIRED = True

# サインアップにメールアドレス認証を挟むか
# ユーザー仮登録→メール送信→リンククリック→ユーザー本登録
ACCOUNT_EMAIL_VERIFICATION = 'none'
ACCOUNT_EMAIL_REQUIRED = False

# ログイン/ログアウト後の遷移先を設定
LOGIN_REDIRECT_URL = 'reversi:index'
ACCOUNT_LOGOUT_REDIRECT_URL = '/accounts/login/'

# ログアウトの挙動
# ログアウトリンクのクリックでログアウトできる
# Falseはログアウトリンククリック→ログアウト画面→ログアウトボタンクリック
ACCOUNT_LOGOUT_ON_GET = True

# Password validation
# https://docs.djangoproject.com/en/2.2/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]


# Internationalization
# https://docs.djangoproject.com/en/2.2/topics/i18n/

LANGUAGE_CODE = 'en-us'

TIME_ZONE = 'UTC'

USE_I18N = True

USE_L10N = True

USE_TZ = True


STATIC_URL = '/static/'