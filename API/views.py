from django.shortcuts import render
from django.http import HttpResponse
from rest_framework.decorators import api_view
from rest_framework.response import Response
from .GptxAlpaca import Alpaca

# Create your views here.
@api_view()
def API(request):
    query=request.GET.get('Question')
    if query is not None:
        model = Alpaca()
        response=model.respond(query)
        return Response({"Answer":response})
    else :
        return Response({"Answer":None})