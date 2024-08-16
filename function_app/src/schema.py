import json

from pydantic import BaseModel as PydanticBaseModel


class LLMResponseBaseModel(PydanticBaseModel):
    """
    A Base Model class that can easily convert a defined Pydantic model into
    a prompt-friendly string representation.
    """

    @classmethod
    def get_prompt_json_example(
        cls, include_preceding_json_instructions: bool = True
    ) -> str:
        model_json_schema = cls.model_json_schema()
        # Add preceding JSON format instructions
        if include_preceding_json_instructions:
            example_response_str = (
                "You are required to return your result as a parsable JSON object, "
                "adhering to the expected format. An example is provided follows:\n"
            )
        else:
            example_response_str = ""
        # Add the JSON object example
        example_response_str = example_response_str + "{\n"
        for field, details in model_json_schema["properties"].items():
            line_str = f""""{field}": {json.dumps(details['examples'][0])}, # {details['description']}"""
            example_response_str += "  " + line_str + "\n"
        example_response_str += "}"
        return example_response_str
