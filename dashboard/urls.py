from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('rooms/', views.rooms, name='rooms'),
    path('analytics/', views.analytics, name='analytics'),
    path('energy/', views.energy, name='energy'),
    path('alerts/', views.alerts, name='alerts'),
    path('settings/', views.settings_view, name='settings'),
    path('simulator/', views.simulator, name='simulator'),
    path('api/v1/rooms/<int:room_id>/status', views.get_room_status, name='get_room_status'),
    path('api/v1/rooms/<int:room_id>/ac', views.toggle_ac, name='toggle_ac'),
    path('api/v1/update_sensors', views.update_sensors, name='update_sensors'),
]
