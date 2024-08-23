connections = [","]
keywords = [f"{i}" for i in range(10)]

numbers = [str(i) for i in range(-10, 64)]

keywords = [f"'{k}'" for k in keywords]
token_list = keywords + numbers + connections
