from django.shortcuts import render


def test_static(request):

    return render(request, 'test_static.html')