from django.http import HttpResponse
from django.shortcuts import render

# Create your views here.
def index_view(request):

    #return HttpResponse('This is music index')
    return render(request, 'music/index.html')