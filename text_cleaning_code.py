import re
s = "how much for the maple_ $  syr-up? ^$20.99? (That's) {ff} [hfhd] |\ridiculous!!!"
remove_symbol = ['~','`','!','@','#','\$','%', '\^','&','\*','\?',':',';',
                 '_','-','/','\|','\(','\)', '{','}','\[','\]','  ']
for sym in remove_symbol:
    print(sym)
    s=re.sub(sym,'',s)
print(s)