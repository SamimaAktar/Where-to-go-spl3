"""Recommendation URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.conf.urls import url
from django.contrib import admin
from django.urls import path
from Recommender.views import home_view
from Recommender.views import sign_in
from Recommender.views import sign_up
from Recommender.views import recommendation

from django.conf import settings
from django.conf.urls.static import static
from django.contrib.staticfiles.urls import staticfiles_urlpatterns

urlpatterns = [
    url(r'^admin/', admin.site.urls),
    #url(r'^home/', home_view),
    path('home/',home_view,name='home_view'),
    path('sign_in/',sign_in,name='sign_in'),
    path('sign_up/',sign_up,name='sign_up'),
    path('recommendation/',recommendation,name='recommendation')

]
#urlpatterns +=staticfiles_urlpatterns()
