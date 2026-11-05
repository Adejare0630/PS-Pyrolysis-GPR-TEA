# %%
import thermosteam as tmo
from thermosteam import Chemical, Chemicals

_cal2joule = 4.184 # auom('cal').conversion_factor('J')
compounds = Chemicals([])


def add_chemical(ID, ref=None, **data):
    chemical = Chemical(ID, **data) if ref is None else ref.copy(ID, **data)
    compounds.append(chemical)
    return chemical

#### Add solids ####
add_chemical('C', search_ID='C', phase='s')
add_chemical('N2', phase='g')
add_chemical("O2", phase='g')
add_chemical("CO")
add_chemical("H2", phase='g')
ch4 = add_chemical("CH4", phase='g')
ch4.aliases.add("NG")
add_chemical("S")
add_chemical("SO2")
add_chemical("H2S")
add_chemical("NO")
add_chemical("NH3")
add_chemical("C2H4")
h2o = add_chemical("H2O")
h2o.aliases.add('Water')
add_chemical("HF")

  

add_chemical('Ash', search_ID='CaO', phase='s')    
add_chemical('Sand', search_ID='SiO2', phase='s')  
add_chemical ('CO2', search_ID='CO2', phase='g')
add_chemical('HDPE', search_ID='CH2')  
c6h6 = add_chemical('C6H6', search_ID='C6H6')
c6h6.Cn.l.add_method(c6h6.Cn('l', T=278, P=101325))
c6h6.Cn.g.add_method(c6h6.Cn('g', T=278, P=101325), Tmin=-1000)
add_chemical('C10H22',search_ID='C10H22', phase='l')
add_chemical('C8H18', search_ID='C8H18')
add_chemical('C4H8', search_ID='C4H8')
add_chemical('C6H12', search_ID='C6H12')
add_chemical('C8H10', search_ID='C8H10')
c7h8 = add_chemical('C7H8', search_ID='C7H8')
c9h12 = add_chemical('C9H12', search_ID= 'C9H12')
c9h12.copy_models_from(c7h8)
PS = add_chemical("PS", search_ID="C8H8")    
PS.Cn.l.add_method(PS.Cn('l', T=278, P=101325))
PS.Cn.g.add_method(PS.Cn('g', T=278, P=101325), Tmin=-1000)

### Add catalysts ####
zeolite = add_chemical('Zeolite', search_ID='Al2O5Si', phase='s',HHV = 0, Hf = 0)  
catalyst = add_chemical('Catalyst', search_db=False, phase='s',HHV = 0, Hf = 0)  
catalyst.default()
nickel = add_chemical('Nickel_catalyst', search_ID='Nickel', phase='s',HHV = 0, Hf = 0)  
zinc_ox = add_chemical('ZnO', search_ID='ZnO', phase='s',HHV = 0, Hf = 0)  
#ZSM5 = add_chemical('ZSM5', search_ID='ZSM5', phase='s', HHV=0, Hf=0)
zeolite.copy_models_from(zinc_ox)
catalyst.copy_models_from(zinc_ox)



### Add refrigerants ###
Tetrafluoroethane = add_chemical('Tetrafluoroethane', search_ID="Tetrafluoroethane")  
ethane_ref = add_chemical('Ethane', search_ID="Ethane")  
propene_ref = add_chemical('Propene', search_ID="Propene")  
propane_ref = add_chemical('Propane', search_ID="Propane")  
add_chemical('Polyethylene', search_ID='C2H4')  

for c in compounds:
    c.default()

tmo.settings.set_thermo(compounds)

compounds["Zeolite"].Tb = 2000
compounds["ZnO"].Tb = 2000
compounds["Catalyst"].Tb = 2000
# %%
