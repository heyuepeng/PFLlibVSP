#!/bin/bash

# ===============================================================horizontal(mnist)======================================================================
# cd ../system/

## jr=1
## 5s
#python runcloud.py -data driving_fg34 -t 5 -e 50 -d cuda:5
#python main.py -data driving_fg34 -algo Local -t 5 -gr 100 -ls 2 -did 0
#python main.py -data driving_fg34 -algo FedAvg -t 5 -did 0
#python main.py -data driving_fg34 -algo FedProx -t 5 -did 0 -mu 0.000001
#python main.py -data driving_fg34 -algo PerAvg -t 5 -did 0
#python main.py -data driving_fg34 -algo FedRep -t 5 -did 0
#python main.py -data driving_fg34 -algo pFedMe -t 5 -did 0 -lr 0.01 -lrp 0.01 -bt 1 -lam 15
#python main.py -data driving_fg34 -algo Ditto -t 5 -did 1
#python main.py -data driving_fg34 -algo APFL -t 5 -did 1
#python main.py -data driving_fg34 -algo FedFomo -t 5 -did 1 -M 5
#python main.py -data driving_fg34 -algo FedALA -t 5 -did 1
#python main.py -data driving_fg34 -algo FedAWA -t 5 -did 3 -r 1 -p 2
## 10s
#python runcloud.py -data driving10_fg34 -t 5 -e 50 -d cuda:5
#python main.py -data driving10_fg34 -algo Local -t 5 -gr 100 -ls 2 -did 5
#python main.py -data driving10_fg34 -algo FedAvg -t 5 -did 2
#python main.py -data driving10_fg34 -algo FedProx -t 5 -did 2 -mu 0.000001
#python main.py -data driving10_fg34 -algo PerAvg -t 5 -did 2
#python main.py -data driving10_fg34 -algo FedRep -t 5 -did 2
#python main.py -data driving10_fg34 -algo pFedMe -t 5 -did 2 -lr 0.01 -lrp 0.01 -bt 1 -lam 15
#python main.py -data driving10_fg34 -algo Ditto -t 5 -did 3
#python main.py -data driving10_fg34 -algo APFL -t 5 -did 3
#python main.py -data driving10_fg34 -algo FedFomo -t 5 -did 3 -M 5
#python main.py -data driving10_fg34 -algo FedALA -t 5 -did 3
#python main.py -data driving10_fg34 -algo FedAWA -t 5 -did 3 -r 1 -p 2
#
#
## jr=0.1
## 5s
#python main.py -data driving_fg34_jr -algo FedAvg -t 5 -did 4 -rjr true -jr 0.1
#python main.py -data driving_fg34_jr -algo FedProx -t 5 -did 4 -mu 0.000001 -rjr true -jr 0.1
#python main.py -data driving_fg34_jr -algo PerAvg -t 5 -did 4 -rjr true -jr 0.1
#python main.py -data driving_fg34_jr -algo FedRep -t 5 -did 4 -rjr true -jr 0.1
#python main.py -data driving_fg34_jr -algo pFedMe -t 5 -did 4 -lr 0.01 -lrp 0.01 -bt 1 -lam 15 -rjr true -jr 0.1
#python main.py -data driving_fg34_jr -algo Ditto -t 5 -did 5 -rjr true -jr 0.1
#python main.py -data driving_fg34_jr -algo APFL -t 5 -did 5 -rjr true -jr 0.1
#python main.py -data driving_fg34_jr -algo FedFomo -t 5 -did 5 -M 5 -rjr true -jr 0.1
#python main.py -data driving_fg34_jr -algo FedALA -t 5 -did 5 -rjr true -jr 0.1
#python main.py -data driving_fg34_jr -algo FedAWA -t 5 -did 3 -r 1 -p 2 -rjr true -jr 0.1
## 10s
#python main.py -data driving10_fg34_jr -algo FedAvg -t 5 -did 6 -rjr true -jr 0.1
#python main.py -data driving10_fg34_jr -algo FedProx -t 5 -did 6 -mu 0.000001 -rjr true -jr 0.1
#python main.py -data driving10_fg34_jr -algo PerAvg -t 5 -did 6 -rjr true -jr 0.1
#python main.py -data driving10_fg34_jr -algo FedRep -t 5 -did 6 -rjr true -jr 0.1
#python main.py -data driving10_fg34_jr -algo pFedMe -t 5 -did 6 -lr 0.01 -lrp 0.01 -bt 1 -lam 15 -rjr true -jr 0.1
#python main.py -data driving10_fg34_jr -algo Ditto -t 5 -did 7 -rjr true -jr 0.1
#python main.py -data driving10_fg34_jr -algo APFL -t 5 -did 7 -rjr true -jr 0.1
#python main.py -data driving10_fg34_jr -algo FedFomo -t 5 -did 7 -M 5 -rjr true -jr 0.1
#python main.py -data driving10_fg34_jr -algo FedALA -t 5 -did 7 -rjr true -jr 0.1
#python main.py -data driving10_fg34_jr -algo FedAWA -t 5 -did 3 -r 1 -p 1 -rjr true -jr 0.1
