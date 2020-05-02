import re
s = "how much for the maple $  syrup? ^$20.99? (That's) {ff} [hfhd] |\ridiculous!!!"
remove_symbol = ['~','`','!','@','#','\$','%', '\^','&','\*','\?',':',';', '/','\|','\(','\)', '{','}','\[','\]','  ']
for sym in remove_symbol:
    print(sym)
    s=re.sub(sym,'',s)
print(s)