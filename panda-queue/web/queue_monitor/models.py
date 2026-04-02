from django.db import models
 
class QueueReading(models.Model):
    timestamp = models.DateTimeField()
    people_count = models.IntegerField()
    estimated_wait_seconds = models.FloatField()
    service_rate = models.FloatField(default=50.0)
 
    class Meta:
        ordering = ['-timestamp']
        indexes = [models.Index(fields=['-timestamp'])]
 
    def __str__(self):
        return (f'{self.timestamp:%H:%M}: '
                f'{self.people_count} people, '
                f'{self.estimated_wait_seconds:.0f}s')