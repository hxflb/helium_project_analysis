import requests
import csv
from bs4 import BeautifulSoup


def init(url, filename, header):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36 Edg/126.0.0.0"
    }
    response = requests.get(url, headers=headers)
    html = response.text
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", attrs={"id": "pro-list-table"})
    file = open(filename, mode="w", encoding='utf-8', newline='')
    csvwriter = csv.writer(file)
    csvwriter.writerow(header)
    trs = table.findAll("tr")[1:]
    return trs, csvwriter, file


url1 = "https://prosettings.net/lists/cs2/"
header1 = ('Team', 'Player', 'Mouse', 'HZ', 'DPI', 'Sens', 'eDPI', 'ZoomSens', 'Monitor', 'Resolution', 'AspectRatio', 'Mousepad', 'Keyboard')
trs1, csvwriter1, f1 = init(url1, "CS2 Pro Settings and Gear List.csv", header1)
for tr in trs1:
    tds = tr.findAll("td")
    Team = tds[1].text
    Player = tds[2].find("span").text
    Mouse = tds[4].text
    HZ = tds[5].text
    DPI = tds[6].text
    Sens = tds[7].text
    eDPI = tds[8].text
    ZoomSens = tds[9].text
    Monitor = tds[10].text
    Resolution = tds[11].text
    AspectRatio = tds[12].text
    Mousepad = tds[14].text
    Keyboard = tds[15].text
    csvwriter1.writerow(
        [Team, Player, Mouse, HZ, DPI, Sens, eDPI, ZoomSens, Monitor, Resolution, AspectRatio, Mousepad, Keyboard])
f1.close()

url2 = "https://prosettings.net/lists/league-of-legends/"
header2 = ('Team', 'Player', 'Resolution', 'Mouse', 'Mousepad', 'Keyboard', 'Headset', 'Monitor')
trs2, csvwriter2, f2 = init(url2, "LOL Pro Settings and Gear List.csv", header2)
for tr in trs2:
    tds = tr.findAll("td")
    Team = tds[1].text
    Player = tds[2].find("span").text
    Resolution = tds[6].text
    Mouse = tds[7].text
    Mousepad = tds[8].text
    Keyboard = tds[9].text
    Headset = tds[10].text
    Monitor = tds[11].text
    csvwriter2.writerow(
        [Team, Player, Resolution, Mouse, Mousepad, Keyboard, Headset, Monitor])
f2.close()

url3 = "https://prosettings.net/lists/valorant/"
header3 = ('Team', 'Player', 'Mouse', 'HZ', 'DPI', 'Sens', 'eDPI', 'ScopedSens', 'Monitor', 'Resolution', 'Mousepad', 'Keyboard', 'Headset')
trs3, csvwriter3, f3 = init(url3, "VALORANT Pro Settings and Gear List.csv", header3)
for tr in trs3:
    tds = tr.findAll("td")
    Team = tds[1].text
    Player = tds[2].find("span").text
    Mouse = tds[3].text
    HZ = tds[4].text
    DPI = tds[5].text
    Sens = tds[6].text
    eDPI = tds[7].text
    ScopedSens = tds[8].text
    Monitor = tds[9].text
    Resolution = tds[10].text
    Mousepad = tds[11].text
    Keyboard = tds[12].text
    Headset = tds[13].text
    csvwriter3.writerow(
        [Team, Player, Mouse, HZ, DPI, Sens, eDPI, ScopedSens, Monitor, Resolution, Mousepad, Keyboard, Headset])
f3.close()

url4 = "https://prosettings.net/lists/apex-legends/"
header4 = ('Team', 'Player', 'Mouse', 'DPI', 'HZ', 'Sens', 'SensMultiplier', 'FOV', 'Monitor', 'Resolution', 'Mousepad', 'Keyboard', 'Headset')
trs4, csvwriter4, f4 = init(url4, "Apex Pro Settings and Gear List.csv", header4)
for tr in trs4:
    tds = tr.findAll("td")
    Team = tds[1].text
    Player = tds[2].find("span").text
    Mouse = tds[3].text
    DPI = tds[4].text
    HZ = tds[5].text
    Sens = tds[6].text
    SensMultiplier = tds[7].text
    FOV = tds[8].text
    Monitor = tds[9].text
    Resolution = tds[10].text
    Mousepad = tds[11].text
    Keyboard = tds[12].text
    Headset = tds[13].text
    csvwriter4.writerow(
        [Team, Player, Mouse, DPI, HZ, Sens, SensMultiplier, FOV, Monitor, Resolution, Mousepad, Keyboard, Headset])
f4.close()
