from django.conf.urls import url
from django.views.decorators.cache import cache_page

from . import views

urlpatterns = [
    url(r'^/catalogs/(?P<catalog_id>\d+)$', cache_page(600,cache='goods')(views.GoodsListView.as_view())),
    url(r'^/detail/(?P<sku_id>\d+)$', views.GoodsDetailView.as_view()),
    url(r'^/index$', cache_page(600,cache='goods')(views.GoodsIndexView.as_view())),
    url(r'^/search$', views.GoodsSearchView.as_view()),
    url(r'^/sku$', views.GoodsChangeSkuView.as_view())
]
