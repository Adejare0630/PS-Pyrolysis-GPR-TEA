#%%# python generate random colors
import pandas as pd
# pyperclip.copy(draw_io(sys))

# install plotly
import plotly.graph_objects as go
import matplotlib.pyplot as plt
def sankeys(sys, scenario="", limit=0, filename="mass_balance_sankey_full.png"):
    links = []

    i = 0
    o = 0
    for s in sys.streams:
        if (s.F_mass-s.imass["H2O"]) < limit: continue
        if s.source == None: # input stream
            # links.append([s.ID, f"{i}", s.sink.ID, s.F_mass])
            links.append([s.ID, s.ID, s.sink.ID, s.F_mass-s.imass["H2O"]])
            i += 1
        elif s.sink == None:
            links.append([s.ID, s.source.ID, s.ID, s.F_mass-s.imass["H2O"]])
            o+=1
        else:
            links.append([s.ID, s.source.ID, s.sink.ID, s.F_mass-s.imass["H2O"]])

    # labels = [u.ID for u in sys.units]
    labels = []
    for l in links:
        labels.append(l[1])
        labels.append(l[2])
    labels = list(set(labels))

    sources = [labels.index(i[1]) for i in links]
    targets = [labels.index(i[2]) for i in links]
    values = [i[3] for i in links]

    link_colors = []
    for v in values:
        alpha = 0.8*v / max(values)+0.2
        link_colors.append(f"rgba(0, 0, 255, {alpha})")

    nodes, sources, targets, values = collapse_chain_nodes(labels, sources, targets, values)

    # create sankey diagram
    fig = go.Figure(data=[go.Sankey(
        node = dict(
            pad = 30,
            thickness = 10,
            line = dict(color = "black", width = 0.5),
            label = labels,
        ),
        link = dict(
            source = sources,
            target = targets,
            value = values,
            color = link_colors
        )  
    )])
    fig.update_layout(title_text=f"{scenario}", font_size=12, width=1000, height=800)
    # fig.write_image(f"{filename}", scale=1, width=1200, height=800)
    fig.show()
# %%

def sankeys_energy(sys, scenario="", limit=0, filename="energy_balance_sankey_full.png"):
    links = []

    i = 0
    o = 0
    for s in sys.streams:
        if s.HHV < limit: continue
        if s.source == None: # input stream
            # links.append([s.ID, f"{i}", s.sink.ID, s.F_mass])
            links.append([s.ID, s.ID, s.sink.ID, s.HHV])
            i += 1
        elif s.sink == None:
            links.append([s.ID, s.source.ID, s.ID, s.HHV])
            o+=1
        else:
            links.append([s.ID, s.source.ID, s.sink.ID, s.HHV])

    for u in sys.units:
        try:
            if u.power_utility.rate*3600 > limit:
                links.append(["W_"+u.ID, "W_"+u.ID, u.ID, u.power_utility.rate*3600])
        except:
            pass

        try:
            if u.Hnet > limit:
                links.append(["Q_"+u.ID, u.ID, "Q_"+u.ID, -u.Hnet])
        except:
            pass

    # labels = [u.ID for u in sys.units]
    labels = []
    for l in links:
        labels.append(l[1])
        labels.append(l[2])
    labels = list(set(labels))

    sources = [labels.index(i[1]) for i in links]
    targets = [labels.index(i[2]) for i in links]
    values = [i[3] for i in links]

    nodes, sources, targets, values = collapse_chain_nodes(labels, sources, targets, values)

    link_colors = []
    for v in values:
        alpha = 0.8*v / max(values)+0.2
        link_colors.append(f"rgba(0, 0, 255, {alpha})")

    # create sankey diagram
    fig = go.Figure(data=[go.Sankey(
        node = dict(
            pad = 30,
            thickness = 10,
            line = dict(color = "black", width = 0.5),
            label = labels,
        ),
        link = dict(
            source = sources,
            target = targets,
            value = values,
            color = link_colors
        )
    )])
    fig.update_layout(title_text=scenario, font_size=12, width=1000, height=800)
    # fig.write_image(filename, scale=1, width=1200, height=800)
    fig.show()
    
    # print(f"Energy Balance: {sum(values)} kJ/hr")
# %%

def collapse_chain_nodes(nodes, source, target, values):
    # Create DataFrame for easy manipulation
    df = pd.DataFrame({"source": source, "target": target, "value": values})
    
    # Identify nodes with only one incoming and one outgoing connection
    single_pass_nodes = [
        node for node in set(df["target"]) 
        if (df["target"] == node).sum() == 1 and (df["source"] == node).sum() == 1
    ]
    
    # Iterate over collapsible nodes and merge them
    for node in single_pass_nodes:
        # Find the single incoming and outgoing connections
        incoming = df[df["target"] == node].iloc[0]
        outgoing = df[df["source"] == node].iloc[0]
        
        # Create a new direct connection
        new_source = incoming["source"]
        new_target = outgoing["target"]
        new_value = min(incoming["value"], outgoing["value"])  # Preserve min flow
        
        # Remove old connections and add new one
        df = df[~df["source"].isin([node]) & ~df["target"].isin([node])]
        df = pd.concat([df, pd.DataFrame([{"source": new_source, "target": new_target, "value": new_value}])], ignore_index=True)
    
    # Update source, target, and value lists
    new_source = df["source"].tolist()
    new_target = df["target"].tolist()
    new_values = df["value"].tolist()
    
    # Update nodes list (remove collapsed nodes)
    new_nodes = [node for node in nodes if node not in single_pass_nodes]
    
    return new_nodes, new_source, new_target, new_values

##################### OLD CODE #####################

# # Energy Balance Sankey Diagram
# endUnits = [u.ID for u in system.flowsheet.unit["FiberDryer"].path_from(
#     system.flowsheet.unit["TFF1"], inclusive=True
# )] + ["FiberDryer"]
# units = [u for u in system.units[0:] if u.ID not in endUnits]
# links = []
# values = []
# colors = []
 
# for u in units:
#     try:
#         if u.power_utility.rate * 3600 / 1000 > 100:
#             links.append([
#                 f"W_{u.ID}: {u.power_utility.rate*3600/1000:.2f} MJ/hr",
#                 u.ID
#             ])
#             values.append(u.power_utility.rate*3600/1000)
#             # blue
#             colors.append("#1f77b4")
#     except:
#         pass
 
#     try:
#         if u.Hnet/1000 > 100:
#             links.append([
#                 f"Q_{u.ID}: {u.Hnet/1000:.2f} MJ/hr",
#                 u.ID
#             ])
#             values.append(u.Hnet/1000)
#             # orange
#             colors.append("#ff7f0e")
#     except:
#         pass
 
#     for i in u.ins:
#         if i.HHV/1000 > 100:
#             if i.source and i.source.ID in [u.ID for u in units]:
#                 try:
#                     links.append([
#                         f"{i.source.ID}",
#                         u.ID
#                     ])
#                     values.append(i.HHV / 1000)
#                     # light gray
#                     colors.append("#d1d1d1")
#                 except:
#                     pass
#             else:
#                 try:
#                     links.append([
#                         f"{i.ID}: {i.HHV/1000:.2f} MJ/hr",
#                         u.ID
#                     ])
#                     values.append(i.HHV/1000)
#                     # light gray
#                     colors.append("#d1d1d1")
#                 except:
#                     pass
 
# for o in system.products:
#     try:
#         if o.source.ID in [u.ID for u in units]:
#             print(o.source.ID, o.ID)
#             links.append([
#                 f"{o.source.ID}",
#                 f"{o.ID}: {o.HHV/1000:.2f} MJ/hr"
#             ])
#             values.append(o.HHV/1000)
#             colors.append("#d1d1d1")
#     except Exception as e:
#         print(f"Could not add {o.ID} to the Sankey diagram.", e)
 
# links.append([
#     "Centrifuge4",
#     f"Fiber: {system.flowsheet.unit['Centrifuge4'].outs[0].HHV/1000:.2f} MJ/hr"
# ])
# values.append(system.flowsheet.unit["Centrifuge4"].outs[0].HHV/1000)
# colors.append("#d1d1d1")
 
# node_labels = list(set([l[0] for l in links] + [l[1] for l in links]))
# source_nodes = [node_labels.index(l[0]) for l in links]
# target_nodes = [node_labels.index(l[1]) for l in links]
 
 
# # pads = [random.randint(15, 100) for _ in range(len(node_labels))]
 
# import random
# def get_color(node):
#     if "W_" in node:
#         return "#1f77b4"
#     elif "Q_" in node:
#         return "#ff7f0e"
#     else:
#         # return a random color for the rest
#         return "rgba(" + ",".join([str(random.randint(0, 255)) for _ in range(3)]) + ", 0.5)"
 
# node_colors = [get_color(node) for node in node_labels]
# fig = go.Figure(
#     go.Sankey(
#         arrangement="snap",
#         node=dict(
#             pad=200000,
#             thickness=10,
#             line=dict(color="black", width=0.5),
#             label=node_labels,
#             color=node_colors,
#             align="right",
#             # color="rgba(31, 119, 180, 0.5)"
#         ),
#         link=dict(
#             source=source_nodes,
#             target=target_nodes,
#             value=values,
#             # label= [v for v in values],
#             color=colors,
#         ),
#     )
# )
 
# fig.update_layout(
#     title_text="Energy Balance Sankey Diagram for Synthetic Spider Silk Process",
#     font_size=14,
#     width=72 * 15,
#     height=72 * 12,
# )
# fig.write_image("energy_balance_plotly.png")
# fig.show()
 
# # %%
# # %%
# # Mass Balance Sankey Diagram
# endUnits = [
#     u.ID
#     for u in system.flowsheet.unit["FiberDryer"].path_from(
#         system.flowsheet.unit["TFF1"], inclusive=True
#     )
# ] + ["FiberDryer"]
# units = [u for u in system.units[0:] if u.ID not in endUnits]
# links = []
# values = []
# colors = []
 
# for u in units:
#     for i in u.ins:
#         if i.F_mass > 1:
#             if i.source and i.source.ID in [u.ID for u in units]:
#                 try:
#                     links.append([f"{i.source.ID}", u.ID])
#                     values.append(i.F_mass)
#                     # light gray
#                     colors.append("#d1d1d1")
#                 except:
#                     pass
#             else:
#                 try:
#                     links.append([f"{i.ID}: {i.F_mass:.2f} kg/hr", u.ID])
#                     values.append(i.F_mass)
#                     # light gray
#                     colors.append("#d1d1d1")
#                 except:
#                     pass
 
# for o in system.products:
#     try:
#         if o.source.ID in [u.ID for u in units]:
#             print(o.source.ID, o.ID)
#             links.append([f"{o.source.ID}", f"{o.ID}: {o.F_mass:.2f} kg/hr"])
#             values.append(o.F_mass)
#             colors.append("#d1d1d1")
#     except Exception as e:
#         print(f"Could not add {o.ID} to the Sankey diagram.", e)
 
# links.append(
#     [
#         "Centrifuge4",
#         "Fiber: "
#         + f"{system.flowsheet.unit['Centrifuge4'].outs[0].F_mass:0.2f}"
#         + " kg/hr",
#     ]
# )
# values.append(system.flowsheet.unit["Centrifuge4"].outs[0].F_mass)
# colors.append("#d1d1d1")
 
# node_labels = list(set([l[0] for l in links] + [l[1] for l in links]))
# source_nodes = [node_labels.index(l[0]) for l in links]
# target_nodes = [node_labels.index(l[1]) for l in links]
 
 
# # pads = [random.randint(15, 100) for _ in range(len(node_labels))]
 
# import random
 
 
# def get_color(node):
#     if "W_" in node:
#         return "#1f77b4"
#     elif "Q_" in node:
#         return "#ff7f0e"
#     else:
#         # return a random color for the rest
#         return (
#             "rgba("
#             + ",".join([str(random.randint(0, 255)) for _ in range(3)])
#             + ", 0.5)"
#         )
 
 
# node_colors = [get_color(node) for node in node_labels]
# fig = go.Figure(
#     go.Sankey(
#         arrangement="snap",
#         node=dict(
#             pad=200,
#             thickness=10,
#             line=dict(color="black", width=0.5),
#             label=node_labels,
#             color=node_colors,
#             align="left",
#             # color="rgba(31, 119, 180, 0.5)"
#         ),
#         link=dict(
#             source=source_nodes,
#             target=target_nodes,
#             value=values,
#             # label= [v for v in values],
#             color=colors,
#         ),
#     )
# )
 
# fig.update_layout(
#     title_text="Mass Balance Sankey Diagram for Synthetic Spider Silk Process",
#     font_size=14,
#     width=72 * 15,
#     height=72 * 15,
# )
# fig.write_image("mass_balance_plotly.png")
# fig.show()
 
 
# # %%
# inChems = []
# for i in system.flowsheet.unit["Dialysis"].ins:
#     for c in i.chemicals:
#         if i.imass[c.ID] > 1:
#             print(c.ID,  i.imass[c.ID], c.HHV, c.HHV*i.imass[c.ID]/1000)
#             inChems.append([c.ID, i.imass[c.ID], c.HHV, c.HHV * i.imass[c.ID] / 1000])
 
# print("\nOuts")
# outs = []
# for o in system.flowsheet.unit["Dialysis"].outs:
#     for c in o.chemicals:
#         if o.imass[c.ID] > 1:
#             print(c.ID, o.imass[c.ID], c.HHV, c.HHV*o.imass[c.ID]/1000)
#             outs.append([c.ID, o.imass[c.ID], c.HHV, c.HHV*o.imass[c.ID]/1000])
# # %%
# sorted(inChems, key=lambda x: x[3], reverse=True)
# # %%
# sorted(outs, key=lambda x: x[3], reverse=True)
 
 