from django.shortcuts import render
from django.http import HttpResponse
from django.shortcuts import get_object_or_404, render
# Create your views here.
import time


def upload(request):
    if request.method == 'POST':
        imagePath = '~/flower/media/uploads/' + str(int(time.time() * 1000)) + '.jpg'
        destination = open(imagePath, 'wb+')
        for chunk in request.FILES['image'].chunks():
                destination.write(chunk)
        destination.close()
	
        return HttpResponse("successful")
     else：
	return HttpResponse("Your method is not post.Please use post.Thank you.")
def index(request):
    return HttpResponse("Beihang University!")
