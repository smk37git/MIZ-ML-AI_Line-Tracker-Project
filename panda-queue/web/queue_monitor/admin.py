from django.contrib import admin
from .models import QueueReading
 
@admin.register(QueueReading)
class QueueReadingAdmin(admin.ModelAdmin):
    list_display = ['timestamp', 'people_count',
                    'estimated_wait_seconds', 'service_rate']
    list_filter = ['timestamp']
    ordering = ['-timestamp']