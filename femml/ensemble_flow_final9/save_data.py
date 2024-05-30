import csv
import tabulate

def save_tsv_dataset(data, file):
    csvfile = open(file, 'w', newline='\n')
    tsv_output = csv.writer(csvfile, delimiter='\t')
    tsv_output.writerows(data)
    csvfile.close()


def save_table_dataset(data, file, colname):
    table = tabulate(data, headers=colname, tablefmt="grid", showindex="always")
    f = open(file, 'w')
    writer = csv.writer(f)
    writer.writerow(colname)
    for row in data:
        writer.writerow(row)
    f.close()

def save_table(table, file):
    f = open(file, 'w')
    f.write(table)
