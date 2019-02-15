from map import get_map, process_map

SEASONS= ("summer","fall","winter","spring")
TRAILS= ("white", "brown", "red")


def run():
    for season in SEASONS:
        print("Preprocessing map for", season, "season...")
        map_img = get_map(season)
        print("Preprocssing map for", season, "season complete!\n")
        for trail in TRAILS:
            print("Finding path in", season, "map for", trail, "trail")
            map, cost = process_map(map_img.copy(), season, trail)
            map_name = season + "_" + trail + ".png"
            map.save(map_name)
            print("Process complete!Output in file", map_name)
            print("Total path cost for", season, trail, "trail:", cost, "m\n")


if __name__ == '__main__':
    run()
