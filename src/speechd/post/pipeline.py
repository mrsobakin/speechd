class PostPipeline:
    def __init__(self, *processors):
        self.processors = processors

    def process(self, text: str) -> str:
        for processor in self.processors:
            text = processor.process(text)

        return text
