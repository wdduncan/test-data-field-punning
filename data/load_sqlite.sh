#!/bin/bash

if [ -e ./simple_data.db ] 
then 
	rm simple_data.db 
fi 

sqlite3 ./simple_data.db "VACUUM;" # create empty db

for filename in *.csv; do
    table=$(basename $filename .csv)
    sqlite3 ./simple_data.db -cmd ".mode csv" ".import $filename $table"
    echo "imported $filename into $table"
done

#### keeping legacy code for future reference ######
##!/bin/bash
#
# check for command line arguments
# if [ "$#" -lt 2 ]; then
# 	echo "You must supply a database name as the first argument, and the directory of GOLD data *.dsv as the second argument"
# 	exit 1
# fi
# 
# sqlite3 $1 "VACUUM;" # create empty db
# 
# for filename in $2/*.dsv; do
#     table=$(basename $filename _DATA_TABLE.dsv)
#     #echo "file: $filename table $table"
#     echo "importing $filename into $table"
#     sqlite3 $1 -cmd ".mode tabs" ".import $filename $table"
# done
# 
#sqlite3 .quit


