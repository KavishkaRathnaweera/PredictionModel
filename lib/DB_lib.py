import mysql.connector
import pandas as pd


def insert_table_stream(table_name, D):
    try:
        mydb = mysql.connector.connect(
            host="localhost",
            user="root",
            password="password",
            database="crypto"
        )
        cursor = mydb.cursor()
        query = f"insert into {table_name} values({D['k']['t']},{D['k']['T']},{D['k']['o']},{D['k']['c']},{D['k']['h']},{D['k']['l']}, {D['k']['v']},{D['k']['n']},{D['k']['q']},{D['k']['V']},{D['k']['Q']})"
        cursor.execute(query)
        mydb.commit()
    except:
        print("There is a problem with MySQL")


"""
 columns = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_taker_ast_vol', 'no_of_trades'
         , 'buy_base_ast_vol', 'taker_buy_quote_ast_vol','ignore']
"""
def insert_table_historical(table_name, D):
    try:
        mydb = mysql.connector.connect(
            host="localhost",
            user="root",
            password="password",
            database="crypto"
        )
        cursor = mydb.cursor()
        for i in D:
            query = f"insert into {table_name} values({i[0]},{i[6]},{i[1]},{i[4]},{i[2]},{i[3]},{i[5]},{i[8]},{i[7]},{i[9]},{i[10]})"
            cursor.execute(query)
        mydb.commit()
    except:
        print("There is a problem with MySQL")


def get_raw_table(symbol, interval):
    try:
        mydb = mysql.connector.connect(
            host="localhost",
            user="root",
            password="password",
            database="crypto"
        )
        cursor = mydb.cursor(dictionary=True)
        query = f"SELECT m01_tab_s FROM m01_mapping WHERE m01_symbol = '{symbol}' AND m01_interval = '{interval}';"
        cursor.execute(query)
        result = cursor.fetchone()
        if result:
            return result['m01_tab_s']
        else:
            return None
    except Exception as e:
        print(f"There is a problem with MySQL table name: {e}")
        return None


def get_last_timestamp(symbol, interval):
    try:
        table_name = get_raw_table(symbol, interval)
        print(table_name)
        mydb = mysql.connector.connect(
            host="localhost",
            user="root",
            password="password",
            database="crypto"
        )
        cursor = mydb.cursor(dictionary=True)
        query = f"SELECT MAX(start_time) as start_time FROM {table_name};"
        cursor.execute(query)
        result = cursor.fetchone()
        if result and result['start_time']:
            return result['start_time']
        else:
            return None
    except Exception as e:
        print(f"There is a problem with MySQL last event time: {e}")
        return None


def get_from_db(query):
    mydb = mysql.connector.connect(
        host="localhost",
        user="root",
        password="password",
        database="crypto"
    )
    cursor = mydb.cursor()
    cursor.execute(query)
    result = cursor.fetchall()
    data_frame = pd.DataFrame(result, columns=cursor.column_names)
    cursor.close()
    mydb.close()
    return data_frame


def get_dataset_for_time_period(symbol, interval, time_steps):
    table_name = get_raw_table(symbol, interval)
    query = f"SELECT * FROM {table_name} ORDER BY start_time DESC LIMIT {time_steps};"
    return get_from_db(query)
