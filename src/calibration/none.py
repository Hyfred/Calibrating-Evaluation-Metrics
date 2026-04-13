class IdentityModel:

    def __call__(self, scores):
        return scores


def train(*args, **kwargs):
    return IdentityModel()


def test(*args, **kwargs):
    data = kwargs["data"]
    model = kwargs["model"]

    return model(data["confidence_score"].to_numpy())
