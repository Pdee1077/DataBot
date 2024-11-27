from django.contrib import admin
from django.urls import path, include  # Import 'include' to link app-specific URLs


urlpatterns = [
    path("admin/", admin.site.urls),  # Admin panel
    path("", include("chatbot_app.urls")),  # Delegate routes to chatbot_app
    
]

# Error handlers
handler404 = "chatbot_app.views.error_404"
handler500 = "chatbot_app.views.error_500"
