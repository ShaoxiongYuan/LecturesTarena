from django.http import HttpResponse

def test_cors(request):

    return HttpResponse('---cors is ok---')