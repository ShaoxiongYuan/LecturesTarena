from django.http import HttpResponse
from django.shortcuts import render

# Create your views here.
def index_view(request):

    #return HttpResponse('This is news index')
    return render(request, 'news/index.html')