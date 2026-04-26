from django.db import models
from django.utils import timezone

class Room(models.Model):
    room_number = models.CharField(max_length=10, unique=True)
    description = models.CharField(max_length=100, default="Deluxe Room")
    floor = models.IntegerField(default=1)
    is_occupied = models.BooleanField(default=False)
    ac_status = models.BooleanField(default=False)

    def __str__(self):
        return f"Room {self.room_number}"

class SensorData(models.Model):
    room = models.ForeignKey(Room, on_delete=models.CASCADE, related_name='sensor_data')
    timestamp = models.DateTimeField(default=timezone.now)
    temperature = models.FloatField()
    gas_ppm = models.FloatField()
    motion_detected = models.BooleanField(default=False)

    def __str__(self):
        return f"{self.room.room_number} - {self.timestamp.strftime('%H:%M:%S')}"

class Alert(models.Model):
    room = models.ForeignKey(Room, on_delete=models.CASCADE, related_name='alerts')
    timestamp = models.DateTimeField(default=timezone.now)
    level = models.CharField(max_length=20, choices=[('NORMAL', 'Normal'), ('WARNING', 'Warning'), ('CRITICAL', 'Critical')], default='NORMAL')
    message = models.TextField()
    meta_info = models.CharField(max_length=200, blank=True, null=True)

    def __str__(self):
        return f"[{self.level}] {self.room.room_number} - {self.message[:20]}"
