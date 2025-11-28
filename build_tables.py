from pyspark.sql import functions as F
from pyspark.sql import types as T

import requests
import gzip
import tarfile
import tempfile
import io
import os


def extrai_info(url, suffix_):
    '''
    Cria um arquivo temporário a partir da extensão do arquivo
    comprimido.

    Extensões comportadas:
    .json.gz
    .gz.csv
    .tar.gz
    '''

    resp = requests.get(url)
    resp.raise_for_status()
    data = resp.content

    if 'json' not in suffix_:
       data = gzip.decompress(data)

    if 'tar' in suffix_:
        tar_file = tarfile.open(fileobj=io.BytesIO(data))
        temp_dir = tempfile.mkdtemp()
        tar_file.extractall(temp_dir)
        tmp_path = os.path.join(temp_dir, os.listdir(temp_dir)[-1])

    else:
      with tempfile.NamedTemporaryFile(suffix=suffix_, delete=False) as tmp:
          tmp.write(data)
          tmp_path = tmp.name

          print('Arquivo temporário criado')

    return tmp_path


def text_to_json(col:str):
  '''
  Trata a informação da coluna items que está em formato string para array.
  '''
  item_schema = item_schema = T.ArrayType(
    T.StructType([
        T.StructField("name", T.StringType()),
        T.StructField("addition", T.StructType([
            T.StructField("value", T.IntegerType()),
            T.StructField("currency", T.StringType()),
        ])),
        T.StructField("discount", T.StructType([
            T.StructField("value", T.IntegerType()),
            T.StructField("currency", T.StringType()),
        ])),
        T.StructField("quantity", T.IntegerType()),
        T.StructField("sequence", T.IntegerType()),
        T.StructField("unitPrice", T.StructType([
            T.StructField("value", T.IntegerType()),
            T.StructField("currency", T.StringType()),
        ])),
        T.StructField("externalId", T.StringType()),
        T.StructField("totalValue", T.StructType([
            T.StructField("value", T.IntegerType()),
            T.StructField("currency", T.StringType()),
        ])),
        T.StructField("customerNote", T.StringType()),
        T.StructField("garnishItems", T.ArrayType(
            T.StructType([
                T.StructField("name", T.StringType()),
                T.StructField("addition", T.StructType([
                    T.StructField("value", T.IntegerType()),
                    T.StructField("currency", T.StringType()),
                ])),
                T.StructField("discount", T.StructType([
                    T.StructField("value", T.StringType()),
                    T.StructField("currency", T.StringType()),
                ])),
                T.StructField("quantity", T.IntegerType()),
                T.StructField("sequence", T.IntegerType()),
                T.StructField("unitPrice", T.StructType([
                    T.StructField("value", T.IntegerType()),
                    T.StructField("currency", T.StringType()),
                ])),
                T.StructField("categoryId", T.StringType()),
                T.StructField("externalId", T.StringType()),
                T.StructField("totalValue", T.StructType([
                    T.StructField("value", T.IntegerType()),
                    T.StructField("currency", T.StringType()),
                ])),
                T.StructField("categoryName", T.StringType()),
                T.StructField("integrationId", T.StringType()),
            ])
        )),
        T.StructField("integrationId", T.StringType()),
        T.StructField("totalAddition", T.StructType([
            T.StructField("value", T.IntegerType()),
            T.StructField("currency", T.StringType()),
        ])),
        T.StructField("totalDiscount", T.StructType([
            T.StructField("value", T.IntegerType()),
            T.StructField("currency", T.StringType()),
        ])),
    ])
)

  return F.from_json(F.col(col), item_schema)



def full_table(orders_df, ab_test_df, consumer_df, restaurant_df):
  '''
  Gera a tabela completa com todas as colunas a considerar em todas as análises.
  '''
  return  (
      orders_df.select(
          F.col('customer_id'),
          F.col('merchant_id'),
          F.col('delivery_address_city'),
          F.col('delivery_address_district'),
          F.col('delivery_address_state'),
          F.col('delivery_address_zip_code'),
          F.date_format(F.col('order_created_at'), 'yyyy-MM-dd').alias('order_date'),
          text_to_json('items').alias('order_items'),
          F.col('order_scheduled'),
          F.col('origin_platform'),
          F.col('order_total_amount')

      ).join(
          consumer_df.select(
              F.col('customer_id'),
              F.col('active'),
              F.col('customer_phone_area')

          ),
          on='customer_id', how='left'
      ).join(
          ab_test_df,
          on='customer_id', how='left'
      ).join(
          restaurant_df.select(
              F.col('id').alias('merchant_id'),
              F.col('enabled'),
              F.col('price_range'),
              F.col('average_ticket'),
              F.col('takeout_time'),
              F.col('delivery_time'),
              F.col('minimum_order_value'),
              F.col('merchant_city'),
              F.col('merchant_state')

          ),
          on='merchant_id', how='left'
      )
  )

def segment_audience(full_df):
  '''
  Gera a segmentação dos clientes baseado no boxplot de frequência de pedidos
  por clientes:

  Percentil 25% - 1. Ped. eventual (<= 1 pedido)
  Mediana - 2. Ped. mensal (<= 1 pedido)
  Percentil 75% - 3. Ped. quinzenal (<= 5 pedido)
  Acima do percentil 75% - 4. Ped. direto (> 5 pedido)
  
  '''
  segmented_user_metrics = (
    full_df.filter(F.col('is_target').isNotNull()

                  ).groupBy('customer_id', 'is_target').agg(
                      F.count('*').alias('totl_order_user')
                    , F.sum('order_total_amount').alias('GMV')
                    # , F.avg('price_range').cast('int').alias('avg_price_range')
                    # , F.avg('average_ticket').cast('int').alias('avg_average_ticket')
                    # , F.avg('takeout_time').cast('int').alias('avg_takeout_time')
                    # , F.avg('delivery_time').cast('int').alias('avg_delivery_time')
                    # , F.avg('minimum_order_value').cast('int').alias('avg_minimum_order_value')

                  ).withColumn('AOV'
                               , F.round(F.col('GMV')/F.col('totl_order_user'), 4)

                  ).withColumn('paid_client', F.round(F.col('GMV')/F.col('totl_order_user'), 4)
                  ).withColumn("order_band",
                                F.when(F.col("totl_order_user") <= 1, "1. Ped. eventual")
                                .when(F.col("totl_order_user") <= 2, "2. Ped. mensal")
                                .when(F.col("totl_order_user") <= 5, "3. Ped. quinzenal")
                                .otherwise("4. Ped. direto")
                            )
                )
  return segmented_user_metrics
