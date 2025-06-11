import os

from openai import OpenAI
from google import genai
from google.genai import types

from dotenv import load_dotenv
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")

gemini_client = genai.Client(api_key=gemini_api_key)

openai_client = OpenAI(api_key=openai_api_key)

def correct_vietnamese_text(text: str) -> str:
    response = openai_client.responses.create(
        model="gpt-4o",
        instructions=(
            "Bạn là chuyên gia ngôn ngữ tiếng Việt. "
            "Hãy sửa đoạn văn sau cho đúng chính tả, đúng ngữ pháp, ngắt nghỉ câu hợp lý (dấu chấm, dấu phẩy, dấu chấm hỏi, v.v.), "
            "và đảm bảo tất cả các chữ cái viết hoa đúng quy tắc tiếng Việt (ví dụ: tên riêng, địa danh, lễ hội). "
            "Chỉ trả về duy nhất đoạn văn đã chỉnh sửa, không thêm lời giải thích."
        ),
        input=text,
    )

    return response.output_text.strip()

def correct_vietnamese_text_gemini(text: str) -> str:
    response = gemini_client.models.generate_content(
        model="gemini-2.0-flash-lite",
        config=types.GenerateContentConfig(
            system_instruction="Bạn là chuyên gia ngôn ngữ tiếng Việt.",
        ),
        contents=f"""
        Hãy sửa đoạn văn sau cho đúng chính tả, đúng ngữ pháp, ngắt nghỉ câu hợp lý (dấu chấm, dấu phẩy, dấu chấm hỏi, v.v.),
        và đảm bảo tất cả các chữ cái viết hoa đúng quy tắc tiếng Việt (ví dụ: tên riêng, địa danh, lễ hội).
        Chỉ trả về duy nhất đoạn văn đã chỉnh sửa, không thêm lời giải thích.
        Đoạn văn:\n{text}
        """
    )
    return response.text

if __name__ == "__main__":
    original_text = (
        "Đây là lần hợp luyện lần thứ 3 là thành phố Ho Chi Minh Sau hơn 2 tháng diễn tập ở các địa phương, "
        "khác với các lần trước Buổi tổng duyệt lần này diễn ra vào ban ngày. tạo điều kiện để các lực lượng "
        "làm quen với điều kiện thực tế. sẵn sàng cho Lễ Diêu Bình. diễu hành chính thức vào sáng ngày 30 tháng 4 năm 2025."
    )
    fixed_text = correct_vietnamese_text_gemini(original_text)
    print(fixed_text)