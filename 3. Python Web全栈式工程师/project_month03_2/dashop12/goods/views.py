from django.core.paginator import Paginator
from django.http import JsonResponse
from django.views.generic import View
import json
from django.conf import settings
from utils.cache_decorator import cache_check
from goods import es_conf
from elasticsearch import Elasticsearch

from .models import *



class GoodsIndexView(View):

    def get(self, request):
        """
        首页商品及分类项展示

        :param result:
        :return:
        """
        # 127.0.0.1:8000/v1/goods/index
        # 0. 获取所有品类
        print('----goods index view in----')
        catalog_list = Catalog.objects.all()
        # 1. 获取各个catalog下的三条sku数据，首页每个品类下面默认显示三个sku
        index_data = []
        # 从redis中获取所有数据

        for cata in catalog_list:
            catalog_dic = {}
            catalog_dic["catalog_id"] = cata.id
            catalog_dic["catalog_name"] = cata.name
            # 1.1 获取拉杆箱sku
            spu_ids = SPU.objects.filter(catalog=cata.id).values("id")
            sku_list = SKU.objects.filter(spu__in=spu_ids, is_launched=True)[:3]
            catalog_dic["sku"] = []
            for sku in sku_list:
                sku_dict = dict()
                sku_dict['skuid'] = sku.id
                sku_dict['caption'] = sku.caption
                sku_dict['name'] = sku.name
                sku_dict['price'] = str(sku.price)
                sku_dict['image'] = str(sku.default_image_url)
                catalog_dic["sku"].append(sku_dict)
            index_data.append(catalog_dic)

        result = {"code": 200, "data": index_data, "base_url": settings.PIC_URL}

        return JsonResponse(result)


class GoodsListView(View):
    def get(self, request, catalog_id):
        """
        获取列表页内容
        :param request:
        :param catalog_id: 分类id
        :param page_num: 第几页
        :param page_size: 每页显示多少项
        :return:
        """
        # 127.0.0.1:8000/v1/goods/catalogs/1/?launched=true&page=1
        # 0. 获取url传递参数值
        launched = bool(request.GET.get('launched', True))
        page_num = request.GET.get('page', 1)
        # 1.获取分类下的sku列表
        spu_list_ids = SPU.objects.filter(catalog=catalog_id).values("id")
        sku_list = SKU.objects.filter(spu__in=spu_list_ids, is_launched=launched).order_by("id")
        # 2.分页
        # 创建分页对象，指定列表、页大小
        page_num = int(page_num)
        page_size = 9
        try:
            paginator = Paginator(sku_list, page_size)
            # 获取指定页码的数据
            page_skus = paginator.page(page_num)
            page_skus_json = []
            for sku in page_skus:
                sku_dict = dict()
                sku_dict['skuid'] = sku.id
                sku_dict['name'] = sku.name
                sku_dict['price'] = str(sku.price)
                sku_dict['image'] = str(sku.default_image_url)
                page_skus_json.append(sku_dict)
        except:
            result = {'code': 40200, 'error': '页数有误，小于0或者大于总页数'}
            return JsonResponse(result)
        result = {'code': 200, 'data': page_skus_json, 'paginator':{'pagesize':page_size, 'total': len(sku_list)}, 'base_url': settings.PIC_URL}
        return JsonResponse(result)


class GoodsDetailView(View):

    @cache_check(key_prefix='gd', key_param='sku_id', cache='goods_detail')
    def get(self, request, sku_id):
        """
        获取sku详情页信息，获取图片暂未完成
        :param request:
        :param sku_id: sku的id
        :return:
        """
        # 127.0.0.1:8000/v1/goods/detail/1
        # 获取sku实例
        print('----goods detail view in----')

        sku_details = {}

        try:
            sku_item = SKU.objects.get(id=sku_id)
        except:
            # 判断是否有当前sku
            result = {'code': 40300, 'error': "Such sku doesn' exist", }
            return JsonResponse(result)
        sku_catalog = sku_item.spu.catalog
        sku_details['image'] = str(sku_item.default_image_url)
        sku_details["spu"] = sku_item.spu.id
        sku_details["name"] = sku_item.name
        sku_details["caption"] = sku_item.caption
        sku_details["price"] = str(sku_item.price)
        sku_details["catalog_id"] = sku_catalog.id
        sku_details["catalog_name"] = sku_catalog.name

        # 详情图片
        sku_image = SKUImage.objects.filter(sku=sku_item)
        if sku_image:
            sku_details['detail_image'] = str(sku_image[0].image)
        else:
            sku_details['detail_image'] = ""

        sku_sale_attr_id = []  # 存放销售属性id
        sku_sale_attr_names = [] # 存放销售属性name
        #spu_sale_attrs = SPUSaleAttr.objects.filter(SPU_id=sku_item.SPU_ID.id).order_by("weight")
        spu_sale_attrs = SPUSaleAttr.objects.filter(spu=sku_item.spu.id)
        for attr in spu_sale_attrs:
            sku_sale_attr_id.append(attr.id)
            sku_sale_attr_names.append(attr.name)


        sku_sale_attr_val_id = [i.id for i in sku_item.sale_attr_value.all()]

        #TODO
        sku_all_sale_attr_vals_id = {} # 存放这个sku所属的spu所有销售属性对应的所有销售属性值id
        sku_all_sale_attr_vals_name = {} # 存放这个sku所属的spu所有销售属性对应的所有销售属性值name
        for id in sku_sale_attr_id:
            items = SaleAttrValue.objects.filter(spu_sale_attr=id)
            sku_all_sale_attr_vals_id[id] = []
            sku_all_sale_attr_vals_name[id] = []
            for item in items:
                sku_all_sale_attr_vals_id[id].append(item.id)
                sku_all_sale_attr_vals_name[id].append(item.name)

        sku_details['sku_sale_attr_id'] = sku_sale_attr_id
        sku_details['sku_sale_attr_names'] = sku_sale_attr_names
        sku_details['sku_sale_attr_val_id'] = sku_sale_attr_val_id
        sku_details['sku_all_sale_attr_vals_id'] = sku_all_sale_attr_vals_id
        sku_details['sku_all_sale_attr_vals_name'] = sku_all_sale_attr_vals_name

        # sku规格部分
        # 用于存放规格相关数据，格式：{规格名称1: 规格值1, 规格名称2: 规格值2, ...}
        spec = dict()
        sku_spec_values = SKUSpecValue.objects.filter(sku=sku_id)
        if not sku_spec_values:
            sku_details['spec'] = dict()
        else:
            for sku_spec_value in sku_spec_values:
                spec[sku_spec_value.spu_spec.name] = sku_spec_value.name
            sku_details['spec'] = spec

        result = {'code': 200, 'data': sku_details, 'base_url': settings.PIC_URL}
        return JsonResponse(result)


# class GoodsSearchView(View):
#     def post(self,request):
#         """
#         首页查询功能
#         :param request:
#         :return:
#         """
#         # 127.0.0.1:8000/v1/goods/search/
#         from dadashop.settings import HAYSTACK_SEARCH_RESULTS_PER_PAGE
#         query = ''
#         page_size = HAYSTACK_SEARCH_RESULTS_PER_PAGE
#         results = EmptySearchQuerySet()
#         if request.POST.get('q'):
#             form = ModelSearchForm(request.POST, searchqueryset=None, load_all=True)
#             if form.is_valid():
#                 query = form.cleaned_data['q']
#                 results = form.search()
#         else:
#             form = ModelSearchForm()
#
#         paginator = Paginator(results, page_size)
#         try:
#             page = paginator.page(int(request.POST.get('page', 1)))
#         except:
#             result = {'code': 40200, 'error': '页数有误，小于0或者大于总页数'}
#             return JsonResponse(result)
#
#         # 记录查询信息
#         context = {
#             'form': form,
#             'page': page,
#             'paginator': paginator,
#             'query': query,
#         }
#
#         sku_list = []
#         # print(len(page.object_list))
#         for result in page.object_list:
#             sku = {
#                 'skuid': result.object.id,
#                 'name': result.object.name,
#                 'price': result.object.price,
#             }
#             # 获取图片
#             sku_image = str(result.object.default_image_url)
#             sku['image'] = sku_image
#             sku_list.append(sku)
#         result = {"code": 200, "data": sku_list, 'paginator': {'pagesize': page_size, 'total': len(results)}, 'base_url': settings.PIC_URL}
#         return JsonResponse(result)


class GoodsSearchView(View):
    def post(self, request):
        """
        2020-3-15 nfx
        es原生API迭代
        首页查询功能
        :param request:
        :return:
        """
        # 127.0.0.1:8000/v1/goods/search
        index_name = es_conf.INDEX_NAME  # 索引名称
        index_type = es_conf.INDEX_TYPE  # 索引类型
        es_ip = es_conf.ES_IP  # es ip
        es = Elasticsearch([es_ip], port=9200, timeout=3600)
        # 分页大小可自定义
        page_size = 9
        query = request.POST.get('q').strip()
        if query:
            # es搜索
            # multi_match:在name和caption里匹配包含query关键字的数据
            body = {
                "query": {
                    "bool": {
                        "should": [
                            {
                                "multi_match": {
                                    "query": query,
                                    "fields": ["name", "caption"]
                                }
                            },
                            {
                                "term": {
                                    "id": query
                                }
                            }
                        ]
                    }
                }
            }

            # 查询name和addr包含 query 关键字的数据
            search_res = es.search(index=index_name, doc_type=index_type, body=body)
        else:
            result = {'code': 40201, 'error': '搜索内容为空，请重新输入'}
            return JsonResponse(result)

        paginator = Paginator(search_res["hits"]["hits"], page_size)
        try:
            page = paginator.page(int(request.POST.get('page', 1)))
        except:
            result = {'code': 40200, 'error': '页数有误，小于0或者大于总页数'}
            return JsonResponse(result)

        sku_list = []
        for result in page.object_list:
            sku_id = result['_source']['id']
            try:
                sku_obj = SKU.objects.get(id=sku_id)
            except Exception as e:
                print(e)
                result = {"code": 40202, "error": "您要查询的商品不存在"}
                return JsonResponse(result)
            sku = {
                'skuid': sku_obj.id,
                'name': sku_obj.name,
                'price': sku_obj.price,
            }
            # 获取图片
            sku_image = str(sku_obj.default_image_url)
            sku['image'] = sku_image
            sku_list.append(sku)
        result = {"code": 200, "data": sku_list, 'paginator': {'pagesize': page_size, 'total': len(search_res["hits"]["hits"])}, 'base_url': settings.PIC_URL}
        return JsonResponse(result)


class GoodsChangeSkuView(View):
    def post(self, request):
        # example = '{"spuid": 1, "1":2, "4": 4}'
        data = json.loads(request.body)
        # 将前端传来的销售属性值id放入列表
        sku_vals = []
        result = {}
        for k in data:
            if 'spuid' != k:
                sku_vals.append(data[k])
        sku_list = SKU.objects.filter(spu=data['spuid'])

        for sku in sku_list:
            sku_details = dict()
            sku_details['sku_id'] = sku.id
            # 获取sku销售属性值id

            sku_sale_attr_val_id = [i.id for i in sku.sale_attr_value.all()]
            #TODO 顺序
            if sku_vals == sku_sale_attr_val_id:
                result = {"code": 200, "data": sku.id,}
        if len(result) == 0:
            result = {"code": 40050, "error": "no such sku",}
        return JsonResponse(result)


# 多表联查
def get_sku_name_and_values(sku_id):
    '''
    多表联查
    :param sku_id:  sku表的id
    :param sku_dict: 存储sku信息的字典
    :return:
    '''
    sku_sale_attr_name = []
    sku_sale_attr_val = []
    # 多表联查

    sales = SKU.objects.filter(id=sku_id).prefetch_related('sale_attr_value__spu_sale_attr')\
        .values('sale_attr_value__name','sale_attr_value__spu_sale_attr__name')
    for sale in sales:
        # 商品属性值
        sku_sale_attr_val.append(sale['sale_attr_value__name'])
        # 商品属性名
        sku_sale_attr_name.append(sale['sale_attr_value__spu_sale_attr__name'])
    return sku_sale_attr_val, sku_sale_attr_name


# 单表查询
# def get_sku_name_and_values(sku_id):
#     '''
#     单表查询
#     :param sku: sku 对象
#     :return:
#     '''
#     sku_sale_attr_name = []
#     sku_sale_attr_val = []
#     # 单表查询
#     sku_obj = SKU.objects.get(id=sku_id)
#     sales = sku_obj.sale_attr_value.all()
#     for sale in sales:
#         sku_sale_attr_val.append(sale.name)
#         sku_sale_attr_name.append(sale.spu_sale_attr.name)
#     return sku_sale_attr_val, sku_sale_attr_name



