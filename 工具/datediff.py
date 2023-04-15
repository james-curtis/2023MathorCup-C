from datetime import datetime

date_format = "%Y-%m-%d"
start_date = datetime.strptime("2021-01-01", date_format)
end_date = datetime.strptime("2022-12-31", date_format)

delta = end_date - start_date
print(delta.days)
