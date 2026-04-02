from django.shortcuts import render

import json
from datetime import timedelta
from django.http import JsonResponse
from django.shortcuts import render
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST, require_GET
from django.utils.dateparse import parse_datetime
from .models import QueueReading
 
def dashboard(request):
    return render(request, 'queue_monitor/dashboard.html')
 
@csrf_exempt
@require_POST
def queue_update(request):
    try:
        data = json.loads(request.body)
        reading = QueueReading.objects.create(
            timestamp=parse_datetime(data['timestamp']) or timezone.now(),
            people_count=data['people_count'],
            estimated_wait_seconds=data['estimated_wait_seconds'],
            service_rate=data.get('service_rate', 50.0),
        )
        return JsonResponse({'status': 'ok', 'id': reading.id})
    except (json.JSONDecodeError, KeyError) as e:
        return JsonResponse({'status': 'error', 'msg': str(e)}, status=400)
 
 
@require_GET
def current_status(request):
    latest = QueueReading.objects.first()
    if not latest:
        return JsonResponse({'people': 0, 'wait_minutes': 0, 'stale': True})
    age = (timezone.now() - latest.timestamp).total_seconds()
    return JsonResponse({
        'people': latest.people_count,
        'wait_minutes': round(latest.estimated_wait_seconds / 60, 1),
        'service_rate': latest.service_rate,
        'timestamp': latest.timestamp.isoformat(),
        'stale': age > 60,
    })
 
 
@require_GET
def history(request):
    hours = int(request.GET.get('hours', 4))
    since = timezone.now() - timedelta(hours=hours)
    readings = list(
        QueueReading.objects.filter(timestamp__gte=since)
        .order_by('timestamp')
        .values('timestamp', 'people_count', 'estimated_wait_seconds')
    )
    return JsonResponse({'data': readings}, default=str)
 
 
@require_GET
def health(request):
    latest = QueueReading.objects.first()
    if not latest:
        return JsonResponse({'status': 'no_data'})
    age = (timezone.now() - latest.timestamp).total_seconds()
    return JsonResponse({
        'status': 'ok' if age < 30 else 'stale',
        'last_update_ago': round(age),
        'total_readings': QueueReading.objects.count(),
    })