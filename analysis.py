from scipy.stats import ttest_ind

from pyspark.sql import functions as F
import numpy as np
import pandas as pd


def cohens_d(a, b):
    '''
    Calcula a diferença das médias pelo desvio padrão
    das duas distribuições.
    Mede impacto da ação
    '''
    cohend = (a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2)
    return cohend


def stat_analysis(date_metrics, col_aval:str):
    '''
    Calcula Cohend para impacto e t-test para significância.
    '''

    df_pd = date_metrics.select('is_target', col_aval).toPandas()
    
    target = df_pd[df_pd.is_target=="target"][col_aval]
    control = df_pd[df_pd.is_target=="control"][col_aval]

    d = cohens_d(target, control)

    t_stat, p_val = ttest_ind(target, control, equal_var=False)

    lenght = '-'*(25-len(col_aval)//2)

    print(lenght, col_aval, lenght)
    print(f"T-stat: {t_stat:.3f}")
    print(f"p-value: {p_val:.3f}")
    print(f"Cohen's d: {d:.3f}")
    print()

    return t_stat, p_val, d

def calculate_user_metrics(full_df):
  user_metrics = (full_df.filter(F.col('is_target').isNotNull()

                  ).groupBy('customer_id', 'is_target').agg(
                      F.count('*').alias('totl_order_user')
                    , F.sum('order_total_amount').alias('GMV')

                  ).withColumn('AOV'
                               , F.round(F.col('GMV')/F.col('totl_order_user'), 4)

                  )
                  )

  camp_metrics = (user_metrics.groupBy('is_target').agg(
                      F.round(F.mean('GMV'), 4).alias('GMV_user')
                      , F.round(F.mean('AOV'), 4).alias('AOV_user')
                      , F.sum('totl_order_user').alias('totl_order')
                      , F.count('*').alias('totl_clients')
                      , F.round(F.sum('GMV'), 4).alias('totl_paid')
                      , F.round(F.mean('totl_order_user'), 4).alias('mean_order_user')
                  )
                            )
  return camp_metrics, user_metrics

  
def calculate_day_metrics(full_df):
  date_metrics = (full_df.filter(F.col('is_target').isNotNull()

                  ).groupBy('order_date', 'is_target').agg(
                      F.count('*').alias('totl_order_day')
                    , F.sum('order_total_amount').alias('GMV')

                  ).withColumn('AOV'
                               , F.round(F.col('GMV')/F.col('totl_order_day'), 4)

                  )
                  )

  return date_metrics

def calculate_dif_audience(camp_metrics, margin:float=0.15, total_days:int=60):

  camp_metrics_pd = camp_metrics.toPandas()

  gmv_control = camp_metrics_pd[camp_metrics_pd['is_target'] == 'control']['totl_paid'].iloc[0]
  gmv_target = camp_metrics_pd[camp_metrics_pd['is_target'] == 'target']['totl_paid'].iloc[0]

  # Calculate incremental GMV
  gmv_incremental = (gmv_target - gmv_control)

  # Calculate Margin
  Margin = gmv_incremental * margin

  print(f"GMV Incremental : R$ {gmv_incremental:,.2f}")
  print(f"GMV Incremental por dia: R$ {gmv_incremental/total_days:,.2f}")
  print(f"Margin (20% de margem): R$ {Margin:,.2f}")
  print(f"Margin (20% de margem) por dia: R$ {Margin/total_days:,.2f}")

  return gmv_incremental, gmv_incremental/total_days, Margin, Margin/total_days

def calculate_dif_segment_audience(segmented_user_metrics, segment:str):
  segmented_metrics_agg = (segmented_user_metrics
                        .groupBy('is_target', segment)
                        .agg(F.sum('GMV').alias('GMV_comp')
                            , F.count('*').alias('clients_comp')
                            , F.sum('totl_order_user').alias('totl_order_comp')
                        ).orderBy(F.col(segment), F.col('is_target')
                        ).withColumn('GMV_user', F.round(F.col('GMV_comp')/F.col('clients_comp'), 4))
                        .withColumn('Pedidos_comp', F.round(F.col('totl_order_comp')/F.col('clients_comp'), 4))
                        .withColumn('AOV_user', F.round(F.col('GMV_comp')/F.col('totl_order_comp'), 4))
)

  segmented_metrics_pd = segmented_metrics_agg.toPandas()


  dif_metrics = {}
  print("Análise de Diferença entre Target e Control por Faixa de Pedidos:")
  for band in segmented_metrics_pd[segment].unique():
      print(f"\n----- {band} -----")
      control_data = segmented_metrics_pd[(segmented_metrics_pd[segment] == band) & (segmented_metrics_pd['is_target'] == 'control')]
      target_data = segmented_metrics_pd[(segmented_metrics_pd[segment] == band) & (segmented_metrics_pd['is_target'] == 'target')]

      if not control_data.empty and not target_data.empty:
          control_gmv_user = control_data['GMV_user'].iloc[0]
          target_gmv_user = target_data['GMV_user'].iloc[0]
          diff_gmv_user = target_gmv_user - control_gmv_user
          pct_diff_gmv_user = (diff_gmv_user / control_gmv_user) * 100 if control_gmv_user != 0 else np.nan

          control_pedidos_comp = control_data['Pedidos_comp'].iloc[0]
          target_pedidos_comp = target_data['Pedidos_comp'].iloc[0]
          diff_pedidos_comp = target_pedidos_comp - control_pedidos_comp
          pct_diff_pedidos_comp = (diff_pedidos_comp / control_pedidos_comp) * 100 if control_pedidos_comp != 0 else np.nan

          control_aov_user = control_data['AOV_user'].iloc[0]
          target_aov_user = target_data['AOV_user'].iloc[0]
          diff_aov_user = target_aov_user - control_aov_user
          pct_diff_aov_user = (diff_aov_user / control_aov_user) * 100 if control_aov_user != 0 else np.nan


          dif_metrics[band] = [diff_gmv_user, diff_pedidos_comp, diff_aov_user]

          print(f"GMV por Usuário (Diferença Absoluta): {diff_gmv_user:,.2f} | (Diferença %): {pct_diff_gmv_user:,.2f}%")
          print(f"Pedidos por Usuário (Diferença Absoluta): {diff_pedidos_comp:,.2f} | (Diferença %): {pct_diff_pedidos_comp:,.2f}%")
          print(f"AOV por Usuário (Diferença Absoluta): {diff_aov_user:,.2f} | (Diferença %): {pct_diff_aov_user:,.2f}%")
      else:
          print("Dados insuficientes para esta banda.")

  return segmented_metrics_pd, dif_metrics
