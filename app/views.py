from django.shortcuts import render
from django.http import HttpResponse
from django.shortcuts import get_object_or_404, render
# Create your views here.
import time


def upload(request):
    if request.method == 'POST':
        filename = request.FILES['image'].name
        imagePath = '/home/ubuntu/flower/media/uploads/' + str(int(time.time() * 1000)) + filename
        destination = open(imagePath, 'wb+')
        for chunk in request.FILES['image'].chunks():
            destination.write(chunk)
        destination.close()
        return HttpResponse("successful")

    else:
        return HttpResponse("Not Post.")


def index(request):
    return HttpResponse("Beihang University!")
