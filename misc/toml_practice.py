import toml

# =============================================================================
# config = toml.load("lol.toml")
# print(config)
# config["batch"]["max_dim"] = 500
# print(config)
# f = open("lol.toml", "w")
# toml.dump(config, f)
# =============================================================================

config = toml.load("lol.toml")
print(config)
config["hello"] = 500
print(config)
f = open("lol.toml", "w")
toml.dump(config, f)
