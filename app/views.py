from django.shortcuts import render
from django.http import HttpResponse
from django.shortcuts import get_object_or_404, render
# Create your views here.
import time
import flower

data = 0
predicted_class = 0

def upload(request):
    if request.method == 'POST':
        filename = request.FILES['image'].name
        imagePath = '/home/ubuntu/flower/media/uploads/' + str(int(time.time() * 1000)) + "-" + filename
        destination = open(imagePath, 'wb+')
        for chunk in request.FILES['image'].chunks():
            destination.write(chunk)
        destination.close()

        width = 50
        model_str = 'model/model.json'
        model_weights = 'model/weights_6.h5'
        global data, predicted_class
        data, predicted_class = flower.model_predict(imagePath, model_str, model_weights, width)

        return HttpResponse(predicted_class)

    else:
        return HttpResponse("No Post!")


def index(request):
    return HttpResponse("Beihang University!")


def getResult(request):
    global data ,predicted_class
    width = 50
    model_str = 'model/model.json'
    model_weights = 'model/weights_6.h5'
    if request.method == 'POST':
        result = request.POST['result']

        flower.feedback_train(data, predicted_class, result, model_str, model_weights, width)
        return HttpResponse("Get your result.")
    else:
        return HttpResponse("No Post!")
