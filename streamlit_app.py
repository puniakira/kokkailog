# -*- coding: utf-8 -*-
# kokalog_gemini_v3_detail_search_streamlit.py

import streamlit as st
import requests
# webbrowser
from urllib.parse import urlencode, quote_plus
import datetime
import json
from collections import defaultdict, Counter
import math
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from dotenv import load_dotenv
import google.generativeai as genai
import time
import pandas as pd

# --- .envファイルから環境変数を読み込む ---
load_dotenv()
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

# --- 定数 ---
ITEMS_PER_PAGE = 10
SPEAKERS_PER_GRAPH_PAGE = 10
MAX_API_RECORDS_SPEECH = 100
MAX_WORKERS = 5
GEMINI_MAX_INPUT_CHARS = 30000
GEMINI_TRUNCATE_METHOD = 'truncate'

# --- Gemini API設定 ---
gemini_safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]
GEMINI_MODEL_NAME = "gemini-1.5-flash-latest"

gemini_init_error = None
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
    except Exception as e:
        gemini_init_error = f"Gemini APIの設定中にエラーが発生しました: {e}"
        GEMINI_API_KEY = None
else:
    gemini_init_error = "Gemini APIキーが設定されていません。環境変数 GEMINI_API_KEY を設定してください。要約/分析/AI検索機能は利用できません。"

# --- APIManager クラス (変更なし) ---
class APIManager:
    def __init__(self):
        pass

    def _make_request(self, url):
        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            content_type = response.headers.get('Content-Type', '')
            if 'application/json' not in content_type:
                error_detail = response.text[:500] if response.text else "(No content)"
                raise ValueError(f"不正な応答形式 (Expected JSON, got {content_type}):\n{error_detail}...")
            data = response.json()
            if "message" in data:
                details = "\n".join(data.get("details", []))
                raise ConnectionError(f"APIエラー: {data['message']}\n{details}")
            return data
        except requests.exceptions.Timeout:
            raise TimeoutError("APIリクエストがタイムアウトしました。")
        except requests.exceptions.HTTPError as e:
             error_detail = e.response.text[:500] if e.response and e.response.text else "(No content)"
             raise ConnectionError(f"APIリクエストエラー (HTTP {e.response.status_code}): {e}\n{error_detail}")
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"APIリクエストエラー: {e}")
        except json.JSONDecodeError:
            raise ValueError(f"API応答のJSON解析に失敗しました。\n応答内容抜粋:\n{response.text[:200]}...")
        except Exception as e:
            raise e

    def fetch_all_speech_data(self, base_params, progress_placeholder):
        all_speech_records = []
        total_records = 0
        is_issue_id_search = 'issueID' in base_params

        status_area = progress_placeholder.empty()
        progress_bar_area = progress_placeholder.empty()

        try:
            if not is_issue_id_search:
                status_area.info("総件数を取得中...")
                first_params = base_params.copy()
                first_params['maximumRecords'] = 1
                first_params['recordPacking'] = 'json'
                url = f"https://kokkai.ndl.go.jp/api/speech?{urlencode(first_params, quote_via=quote_plus)}"
                first_data = self._make_request(url)
                total_records = int(first_data.get("numberOfRecords", 0))
                if total_records == 0:
                    status_area.info("該当データなし")
                    return [], 0
                status_area.info(f"総件数: {total_records}件。データを取得中...")
            else:
                total_records = -1
                status_area.info(f"会議録ID指定でデータを取得中...")

            urls_to_fetch = []
            if not is_issue_id_search and total_records > 0:
                num_requests = math.ceil(total_records / MAX_API_RECORDS_SPEECH)
                for i in range(num_requests):
                    start_record = i * MAX_API_RECORDS_SPEECH + 1
                    req_params = base_params.copy()
                    req_params['maximumRecords'] = MAX_API_RECORDS_SPEECH
                    req_params['startRecord'] = start_record
                    req_params['recordPacking'] = 'json'
                    urls_to_fetch.append(f"https://kokkai.ndl.go.jp/api/speech?{urlencode(req_params, quote_via=quote_plus)}")
            elif is_issue_id_search or (not is_issue_id_search and total_records <= MAX_API_RECORDS_SPEECH and total_records > 0):
                 req_params = base_params.copy()
                 req_params['maximumRecords'] = MAX_API_RECORDS_SPEECH
                 req_params['recordPacking'] = 'json'
                 urls_to_fetch.append(f"https://kokkai.ndl.go.jp/api/speech?{urlencode(req_params, quote_via=quote_plus)}")
            elif total_records == 0:
                 return [], 0

            if not urls_to_fetch:
                 status_area.info("該当データなし")
                 return [], 0

            results = []
            progress_bar = None
            if len(urls_to_fetch) > 0 :
                progress_bar = progress_bar_area.progress(0)

            with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(urls_to_fetch))) as executor:
                futures = {executor.submit(self._make_request, fetch_url): fetch_url for fetch_url in urls_to_fetch}
                processed_count = 0
                total_futures = len(futures)

                for i, future in enumerate(as_completed(futures)):
                    try:
                        data = future.result()
                        current_records = data.get("speechRecord", [])
                        results.append(data)
                        processed_count += len(current_records)

                        if is_issue_id_search and total_records == -1:
                            total_records = int(data.get("numberOfRecords", len(current_records)))

                        if total_records > 0 :
                            progress_percent_api = (processed_count / total_records) * 100 if total_records > 0 else 0
                            status_text = f"データ取得中... ({processed_count}/{total_records}件, {progress_percent_api:.1f}%) - リクエスト {i+1}/{total_futures}"
                        else:
                            status_text = f"データ取得中... ({processed_count}件) - リクエスト {i+1}/{total_futures}"
                        status_area.info(status_text)
                        if progress_bar:
                            progress_bar.progress( (i + 1) / total_futures )

                    except Exception as e:
                        failed_url = futures[future]
                        st.error(f"データ取得中にエラー (URL: ...{failed_url[-50:]}): {e}")

            for data in results:
                all_speech_records.extend(data.get("speechRecord", []))

            status_area.success(f"データ取得完了。全{len(all_speech_records)}件の発言を取得。")
            progress_bar_area.empty()


            unique_speeches = {s['speechID']: s for s in all_speech_records}.values()
            final_total = total_records if total_records != -1 else len(unique_speeches)

            return list(unique_speeches), final_total

        except (ConnectionError, TimeoutError, ValueError, Exception) as e:
            st.error(f"APIデータ取得エラー: {e}")
            status_area.empty()
            progress_bar_area.empty()
            return [], 0

# --- ResultProcessor クラス (変更なし) ---
class ResultProcessor:
    def __init__(self, keywords=None):
        self.keywords = keywords if keywords else {}
        self.all_meetings_grouped = {}
        self.meeting_order = []
        self.total_pages = 1

    def process_speeches(self, speech_records):
        self.all_meetings_grouped = defaultdict(lambda: {"info": {}, "speeches": []})
        temp_meeting_info = {}

        for speech_info in speech_records:
            speech_text = speech_info.get("speech", "")
            if speech_text.strip().startswith("000\u3000") or re.match(r"^\d+\u3000", speech_text.strip()):
                 continue

            meeting_url = speech_info.get("meetingURL")
            issue_id = speech_info.get("issueID")
            if not meeting_url or not issue_id:
                 continue

            if meeting_url not in temp_meeting_info:
                temp_meeting_info[meeting_url] = {
                    "name": speech_info.get("nameOfMeeting", "会議名不明"),
                    "date": speech_info.get("date", ""),
                    "house": speech_info.get("nameOfHouse", ""),
                    "session": speech_info.get("session", ""),
                    "issue": speech_info.get("issue", ""),
                    "issueID": issue_id
                }
                self.all_meetings_grouped[meeting_url]["info"] = temp_meeting_info[meeting_url]

            highlighted_speech_info = self._get_highlight_info(speech_text, 'utterance')
            highlighted_speaker_info = self._get_highlight_info(speech_info.get("speaker", "発言者不明"), 'speaker')

            processed_speech = {
                "speechID": speech_info.get("speechID"),
                "speaker": speech_info.get("speaker", "発言者不明"),
                "speaker_hl_info": highlighted_speaker_info,
                "speech": speech_text,
                "speech_hl_info": highlighted_speech_info,
                "speechURL": speech_info.get("speechURL"),
                "speechOrder": speech_info.get("speechOrder", "")
            }
            self.all_meetings_grouped[meeting_url]["speeches"].append(processed_speech)

        for url in self.all_meetings_grouped:
             try:
                 self.all_meetings_grouped[url]["speeches"].sort(key=lambda x: int(x.get('speechOrder', 0)) if str(x.get('speechOrder', '0')).isdigit() else 0)
             except ValueError:
                 st.caption(f"注意: {url} で数値でないspeechOrderが見つかりました。ソートが不正確になる可能性があります。")

        self.meeting_order = sorted(self.all_meetings_grouped.keys(),
                                    key=lambda url: self.all_meetings_grouped[url]["info"]["date"],
                                    reverse=True)

        self.total_pages = math.ceil(len(self.meeting_order) / ITEMS_PER_PAGE)
        if self.total_pages == 0: self.total_pages = 1
        return len(self.meeting_order) > 0

    def _get_highlight_info(self, text, keyword_type):
        keyword = self.keywords.get(keyword_type, "")
        if not keyword or not text:
            return {"text": text, "ranges": []}
        keywords_to_highlight = keyword.split()
        ranges = []
        for kw in keywords_to_highlight:
            if not kw: continue
            try:
                if re.search(r'[.^$*+?{}\[\]\\|()]', kw):
                    pattern = re.escape(kw)
                    flags = 0
                else:
                    pattern = re.escape(kw)
                    flags = re.IGNORECASE
                for match in re.finditer(pattern, text, flags):
                    ranges.append((match.start(), match.end()))
            except re.error:
                st.caption(f"警告: キーワード '{kw}' の正規表現コンパイルエラー。単純検索にフォールバックします。")
                start = 0
                while True:
                    idx = text.find(kw, start)
                    if idx == -1: break
                    ranges.append((idx, idx + len(kw)))
                    start = idx + 1
        ranges.sort()
        merged_ranges = []
        if ranges:
            current_start, current_end = ranges[0]
            for next_start, next_end in ranges[1:]:
                if next_start < current_end:
                    current_end = max(current_end, next_end)
                else:
                    merged_ranges.append((current_start, current_end))
                    current_start, current_end = next_start, next_end
            merged_ranges.append((current_start, current_end))
        return {"text": text, "ranges": merged_ranges}

    def get_page_data(self, page_num):
        if not (1 <= page_num <= self.total_pages):
            return []
        start_index = (page_num - 1) * ITEMS_PER_PAGE
        end_index = start_index + ITEMS_PER_PAGE
        page_meeting_urls = self.meeting_order[start_index:end_index]
        page_data = []
        for url in page_meeting_urls:
            meeting_data = self.all_meetings_grouped[url]
            highlighted_meeting_name_info = self._get_highlight_info(meeting_data["info"]["name"], 'committee')
            page_data.append({
                "url": url,
                "info": meeting_data["info"],
                "info_hl_info": {"name": highlighted_meeting_name_info},
                "speeches": meeting_data["speeches"]
            })
        return page_data

    def get_meeting_details(self, meeting_url):
        if meeting_url in self.all_meetings_grouped:
            meeting_data = self.all_meetings_grouped[meeting_url]
            combined_text_parts = []
            speech_links = []
            for s in meeting_data['speeches']:
                speech_url = s.get('speechURL', 'N/A')
                combined_text_parts.append(f"[発言者: {s['speaker']} | URL: {speech_url}]\n{s['speech']}")
                if speech_url != 'N/A':
                    speech_links.append(speech_url)
            combined_text = "\n\n---\n\n".join(combined_text_parts)
            return combined_text, speech_links
        return "", []

    def get_speaker_details(self, speaker_name):
        speeches_text_parts = []
        speech_details_list = []
        for url in self.meeting_order:
            meeting_data = self.all_meetings_grouped[url]
            meeting_context = f"({meeting_data['info']['date']} {meeting_data['info']['name']})"
            for speech_info in meeting_data['speeches']:
                if speech_info['speaker'] == speaker_name:
                    speech_url = speech_info.get('speechURL', 'N/A')
                    speeches_text_parts.append(f"[発言者: {speaker_name} @ {meeting_context} | URL: {speech_url}]\n{speech_info['speech']}")
                    snippet = speech_info['speech'][:80].replace('\n', ' ').strip() + "..."
                    speech_details_list.append({
                        'url': speech_url if speech_url != 'N/A' else None,
                        'snippet': snippet,
                        'context': meeting_context,
                        'speaker': speaker_name
                    })
        combined_text = "\n\n---\n\n".join(speeches_text_parts)
        return combined_text, speech_details_list

    def get_all_context_text(self, max_chars=GEMINI_MAX_INPUT_CHARS, truncate_method=GEMINI_TRUNCATE_METHOD):
        all_text_parts = []
        total_len = 0
        truncated = False
        for url in self.meeting_order:
            meeting_data = self.all_meetings_grouped[url]
            meeting_header = f"# 会議: {meeting_data['info']['name']} ({meeting_data['info']['date']})\nMeeting URL (Internal Ref): {url}\n\n"
            if total_len + len(meeting_header) > max_chars:
                 truncated = True
                 break
            all_text_parts.append(meeting_header)
            total_len += len(meeting_header)
            for s in meeting_data['speeches']:
                speech_url = s.get('speechURL', 'N/A')
                speech_line = f"[発言者: {s['speaker']} | 発言順: {s.get('speechOrder', '?')} | URL: {speech_url}]\n{s['speech']}\n---\n"
                if total_len + len(speech_line) > max_chars:
                    if truncate_method == 'truncate':
                         remaining_chars = max_chars - total_len
                         if remaining_chars > 50:
                             all_text_parts.append(speech_line[:remaining_chars] + "...\n[打ち切り]\n")
                             total_len += remaining_chars
                    truncated = True
                    break
                all_text_parts.append(speech_line)
                total_len += len(speech_line)
            if truncated:
                break
        full_text = "".join(all_text_parts)
        if truncated:
            st.caption(f"注意: AI用のコンテキストが {max_chars} 文字で制限されました (Method: {truncate_method})。実際の文字数: {total_len}")
            if truncate_method == 'error':
                 raise ValueError(f"コンテキストが長すぎます。最大許容文字数 {max_chars} を超えました (推定 {total_len} 文字)。検索範囲を狭めてください。")
        return full_text, truncated

    def get_speaker_speech_counts(self, utterance_keyword):
        speaker_counts = Counter()
        if not utterance_keyword:
            return speaker_counts
        keyword_lower = utterance_keyword.lower()
        for meeting_url in self.meeting_order:
            meeting_data = self.all_meetings_grouped[meeting_url]
            for speech_info in meeting_data['speeches']:
                speech_text = speech_info.get('speech', '')
                speaker_name = speech_info.get('speaker', '発言者不明')
                if keyword_lower in speech_text.lower():
                    speaker_counts[speaker_name] += 1
        return speaker_counts

# --- Streamlit UI & Application Logic ---
def initialize_session_state():
    min_date = datetime.date(1947, 1, 1)
    today = datetime.date.today()
    defaults = {
        'search_keywords': {},
        'api_manager': APIManager(),
        'original_speech_data': [],
        'current_display_processor': ResultProcessor(),
        'is_filtered_mode': False,
        'current_page': 1,
        'status_message': "準備完了",
        'gemini_model': None,
        'active_gemini_popup': None,
        'gemini_popup_content': "",
        'gemini_popup_status': "",
        'search_in_progress': False,
        'start_date_val': None,
        'end_date_val': None,
        'session_val': "-",
        'house_val': "-",
        'meeting_type_val': "-",
        'committee_val': "",
        'speaker_val': "",
        'utterance_val': "",
        'filter_keyword_streamlit': "",
        'ai_question_streamlit': "",
        'page_select_key': "1",
        'page_select_key_bottom': "1",
        'min_date_limit': min_date,
        'max_date_limit': today,
        'speaker_speech_counts_df': pd.DataFrame(columns=['発言者', '発言数']),
        # 'graph_sort_order': '多い順', # 削除
        'graph_speaker_page': 1,
        'graph_speaker_page_select_key': "1", # グラフ下発言者リストのページ選択用
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def get_gemini_model_st():
    if not GEMINI_API_KEY: return None
    if st.session_state.gemini_model is None:
        try:
            st.session_state.gemini_model = genai.GenerativeModel(GEMINI_MODEL_NAME, safety_settings=gemini_safety_settings)
        except Exception as e:
            st.error(f"Geminiモデルの初期化に失敗しました: {e}")
            return None
    return st.session_state.gemini_model

def format_with_highlight(text_info):
    text = text_info["text"]
    ranges = sorted(text_info["ranges"], key=lambda x: x[0])
    if not ranges: return text
    result_parts = []
    last_end = 0
    for start, end in ranges:
        if start > last_end: result_parts.append(text[last_end:start])
        if start < end: result_parts.append(f"<mark>{text[start:end]}</mark>")
        last_end = end
    if last_end < len(text): result_parts.append(text[last_end:])
    return "".join(result_parts)

def display_page_streamlit(page_num):
    processor = st.session_state.current_display_processor
    page_data = processor.get_page_data(page_num)

    if not page_data:
        st.info("このページに表示するデータがありません。")
        return

    for meeting_idx, meeting in enumerate(page_data):
        info = meeting["info"]
        info_hl_info = meeting["info_hl_info"]
        meeting_url = meeting["url"]
        meeting_name = info['name']

        meeting_name_md = format_with_highlight(info_hl_info['name'])
        header_text = f"**{meeting_name_md}** ({info['house']} 第{info['session']}回 {info['issue']}号 {info['date']})"

        with st.expander(header_text, expanded=False):
            action_cols = st.columns([1,1,2])
            action_cols[0].link_button("🌐 会議録を開く", meeting_url, help=f"国会会議録検索システムのページを開きます: {meeting_url}", use_container_width=True)
            if GEMINI_API_KEY:
                if action_cols[1].button("📝 要約 (Gemini)", key=f"summary_btn_{meeting_url}_{page_num}_{meeting_idx}", help="この会議の内容をGeminiで要約します。", use_container_width=True, disabled=st.session_state.search_in_progress):
                    handle_meeting_summary(meeting_url, meeting_name)
            else:
                action_cols[1].caption("要約機能無効")

            speech_display_count = 0
            for speech_idx, speech_info in enumerate(meeting["speeches"]):
                if speech_display_count >= 3 and len(meeting["speeches"]) > 3:
                    st.caption("    ...")
                    break
                speaker_name = speech_info["speaker"]
                speaker_md = format_with_highlight(speech_info["speaker_hl_info"])
                speech_text_md = format_with_highlight(speech_info["speech_hl_info"])
                speech_url = speech_info.get("speechURL")

                st.markdown(f"**{speaker_md}**:")
                speech_display_text = speech_text_md
                if speech_url:
                    speech_display_text += f" <small>[発言リンク]({speech_url})</small>"
                st.markdown(speech_display_text, unsafe_allow_html=True)

                speaker_action_cols = st.columns([1,1,2])
                if GEMINI_API_KEY:
                     if st.session_state.utterance_val.strip():
                        if speaker_action_cols[0].button("🔬 分析 (Gemini)", key=f"analyze_btn_{meeting_url}_{speaker_name}_{page_num}_{speech_idx}", help=f"{speaker_name}氏の「{st.session_state.utterance_val.strip()}」に関する発言をGeminiで分析します。", use_container_width=True, disabled=st.session_state.search_in_progress):
                            handle_speaker_analysis(speaker_name)
                else:
                    if st.session_state.utterance_val.strip():
                        speaker_action_cols[0].caption("分析機能無効")

                if speaker_action_cols[1].button("🔗 発言一覧", key=f"links_btn_{meeting_url}_{speaker_name}_{page_num}_{speech_idx}", help=f"{speaker_name}氏の関連発言リンク一覧を表示します。", use_container_width=True, disabled=st.session_state.search_in_progress):
                    handle_speaker_links(speaker_name)
                st.divider()
                speech_display_count += 1

def start_search_streamlit():
    st.session_state.search_in_progress = True
    st.session_state.status_message = "検索条件を収集中..."
    st.session_state.original_speech_data = []
    st.session_state.current_display_processor = ResultProcessor()
    st.session_state.current_page = 1
    st.session_state.is_filtered_mode = False
    st.session_state.filter_keyword_streamlit = ""
    st.session_state.ai_question_streamlit = ""
    st.session_state.speaker_speech_counts_df = pd.DataFrame(columns=['発言者', '発言数'])
    st.session_state.graph_speaker_page = 1
    st.session_state.graph_speaker_page_select_key = "1"


    params = {}
    current_search_keywords = {}

    start_date = st.session_state.start_date_val
    end_date = st.session_state.end_date_val
    if start_date: params['from'] = start_date.strftime('%Y-%m-%d')
    if end_date: params['until'] = end_date.strftime('%Y-%m-%d')
    if start_date and end_date and start_date > end_date:
        st.error("日付範囲エラー: 終了日は開始日以降の日付を指定してください。")
        st.session_state.search_in_progress = False
        return

    session_str = st.session_state.session_val
    if session_str and session_str != "-":
        params['sessionFrom'] = re.sub(r'[第回国会]', '', session_str)
        params['sessionTo'] = params['sessionFrom']

    house = st.session_state.house_val
    if house and house != "-": params['nameOfHouse'] = house

    meeting_type = st.session_state.meeting_type_val
    committee_name = st.session_state.committee_val.strip()
    if committee_name: current_search_keywords['committee'] = committee_name
    meeting_name_value = None
    if meeting_type == "本会議": meeting_name_value = "本会議"
    elif meeting_type == "委員会等" and committee_name: meeting_name_value = committee_name
    elif committee_name: meeting_name_value = committee_name
    if meeting_name_value: params['nameOfMeeting'] = meeting_name_value

    speaker = st.session_state.speaker_val.strip()
    if speaker:
        params['speaker'] = speaker
        current_search_keywords['speaker'] = speaker

    utterance = st.session_state.utterance_val.strip()
    if utterance:
        params['any'] = utterance
        current_search_keywords['utterance'] = utterance

    st.session_state.search_keywords = current_search_keywords

    meaningful_params = {k: v for k, v in params.items() if k not in ['maximumRecords', 'recordPacking', 'startRecord']}
    if not meaningful_params:
         st.warning("検索条件を少なくとも1つ指定してください。")
         st.session_state.search_in_progress = False
         return

    st.session_state.status_message = "API問い合わせ中..."
    progress_placeholder = st.empty()
    try:
        fetched_data, total_api_records = st.session_state.api_manager.fetch_all_speech_data(params, progress_placeholder)
        st.session_state.original_speech_data = fetched_data
        if not st.session_state.original_speech_data:
            st.info("該当する結果が見つかりませんでした。")
            st.session_state.status_message = "結果なし"
        else:
            st.session_state.status_message = "結果を処理中..."
            processor = ResultProcessor(keywords=st.session_state.search_keywords)
            found_meetings = processor.process_speeches(st.session_state.original_speech_data)
            st.session_state.current_display_processor = processor
            if found_meetings:
                st.session_state.status_message = f"{len(processor.meeting_order)}件の会議が見つかりました。"
                st.session_state.current_page = 1
                st.session_state.page_select_key = "1"
                st.session_state.page_select_key_bottom = "1"

                utterance_keyword_for_graph = st.session_state.utterance_val.strip()
                if utterance_keyword_for_graph:
                    speaker_counts = processor.get_speaker_speech_counts(utterance_keyword_for_graph)
                    if speaker_counts:
                        df = pd.DataFrame(speaker_counts.items(), columns=['発言者', '発言数'])
                        df = df.sort_values('発言数', ascending=False) # 多い順でソート
                        st.session_state.speaker_speech_counts_df = df
                    else:
                        st.session_state.speaker_speech_counts_df = pd.DataFrame(columns=['発言者', '発言数'])
            else:
                st.info("該当する結果が見つかりませんでした（処理後）。")
                st.session_state.status_message = "結果なし（処理後）"
    except Exception as e:
        st.error(f"検索処理中にエラーが発生しました: {e}")
        st.session_state.status_message = "検索エラー"
    finally:
        st.session_state.search_in_progress = False
        progress_placeholder.empty()
        st.rerun()

def clear_search_fields_streamlit():
    st.session_state.start_date_val = None
    st.session_state.end_date_val = None
    st.session_state.session_val = "-"
    st.session_state.house_val = "-"
    st.session_state.meeting_type_val = "-"
    st.session_state.committee_val = ""
    st.session_state.speaker_val = ""
    st.session_state.utterance_val = ""
    st.session_state.search_keywords = {}
    st.session_state.status_message = "入力フィールドをクリアしました。"
    st.session_state.original_speech_data = []
    st.session_state.current_display_processor = ResultProcessor()
    st.session_state.current_page = 1
    st.session_state.is_filtered_mode = False
    st.session_state.speaker_speech_counts_df = pd.DataFrame(columns=['発言者', '発言数'])
    st.session_state.graph_speaker_page = 1
    st.session_state.graph_speaker_page_select_key = "1"
    st.rerun()


def filter_results_streamlit():
    filter_keyword = st.session_state.filter_keyword_streamlit.strip()
    if not filter_keyword:
        st.info("絞り込みキーワードを入力してください。")
        return
    if not st.session_state.original_speech_data:
        st.info("絞り込む対象の検索結果がありません。先に通常の検索を実行してください。")
        return

    st.session_state.search_in_progress = True
    st.session_state.status_message = f"結果内検索:「{filter_keyword}」で絞り込み中..."
    st.session_state.speaker_speech_counts_df = pd.DataFrame(columns=['発言者', '発言数'])
    st.session_state.graph_speaker_page = 1
    st.session_state.graph_speaker_page_select_key = "1"


    keyword_lower = filter_keyword.lower()
    filtered_speeches = []
    try:
        for speech_info in st.session_state.original_speech_data:
            match = False
            if keyword_lower in speech_info.get('speaker', '').lower(): match = True
            elif keyword_lower in speech_info.get('speech', '').lower(): match = True
            if match: filtered_speeches.append(speech_info)

        st.session_state.is_filtered_mode = True
        if not filtered_speeches:
            st.session_state.current_display_processor = ResultProcessor()
            st.session_state.current_page = 1
            st.session_state.page_select_key = "1"
            st.session_state.page_select_key_bottom = "1"
            st.info(f"結果内検索「{filter_keyword}」: 0件")
            st.session_state.status_message = f"結果内検索「{filter_keyword}」: 0件"
        else:
            new_processor = ResultProcessor(keywords=st.session_state.search_keywords)
            found_meetings = new_processor.process_speeches(filtered_speeches)
            st.session_state.current_display_processor = new_processor
            st.session_state.current_page = 1
            st.session_state.page_select_key = "1"
            st.session_state.page_select_key_bottom = "1"
            if found_meetings:
                st.session_state.status_message = f"結果内検索「{filter_keyword}」: {len(new_processor.meeting_order)}件の会議"
                utterance_keyword_for_graph = st.session_state.utterance_val.strip()
                if utterance_keyword_for_graph:
                    speaker_counts = new_processor.get_speaker_speech_counts(utterance_keyword_for_graph)
                    if speaker_counts:
                        df = pd.DataFrame(speaker_counts.items(), columns=['発言者', '発言数'])
                        df = df.sort_values('発言数', ascending=False) # 多い順でソート
                        st.session_state.speaker_speech_counts_df = df
            else:
                st.info(f"結果内検索「{filter_keyword}」: 処理後 0件")
                st.session_state.status_message = f"結果内検索「{filter_keyword}」: 処理後 0件"
    except Exception as e: st.error(f"絞り込み処理中にエラー: {e}")
    finally:
        st.session_state.search_in_progress = False
        st.rerun()

def reset_filter_streamlit():
    if not st.session_state.is_filtered_mode and not st.session_state.original_speech_data:
        st.info("元の検索結果がありません。")
        return

    st.session_state.search_in_progress = True
    st.session_state.status_message = "全件表示に戻しています..."
    st.session_state.speaker_speech_counts_df = pd.DataFrame(columns=['発言者', '発言数'])
    st.session_state.graph_speaker_page = 1
    st.session_state.graph_speaker_page_select_key = "1"

    try:
        processor = ResultProcessor(keywords=st.session_state.search_keywords)
        found_meetings = processor.process_speeches(st.session_state.original_speech_data)
        st.session_state.current_display_processor = processor
        st.session_state.current_page = 1
        st.session_state.page_select_key = "1"
        st.session_state.page_select_key_bottom = "1"
        st.session_state.is_filtered_mode = False
        st.session_state.filter_keyword_streamlit = ""
        if found_meetings:
            st.session_state.status_message = f"{len(processor.meeting_order)}件の会議を表示中（全件）"
            utterance_keyword_for_graph = st.session_state.utterance_val.strip()
            if utterance_keyword_for_graph:
                speaker_counts = processor.get_speaker_speech_counts(utterance_keyword_for_graph)
                if speaker_counts:
                    df = pd.DataFrame(speaker_counts.items(), columns=['発言者', '発言数'])
                    df = df.sort_values('発言数', ascending=False) # 多い順でソート
                    st.session_state.speaker_speech_counts_df = df
        else:
            st.info("元のデータを再処理しましたが、表示できる会議がありませんでした。")
            st.session_state.status_message = "全件表示エラー"
    except Exception as e: st.error(f"全件表示への復元中にエラー: {e}")
    finally:
        st.session_state.search_in_progress = False
        st.rerun()

# --- Gemini Helper for Candidate Info (変更なし) ---
def get_candidate_info(response):
    candidate_info_str = ""
    if hasattr(response, 'candidates') and response.candidates:
        for i, cand in enumerate(response.candidates):
            candidate_info_str += f"\n  候補 {i+1}:"
            if hasattr(cand, 'finish_reason') and cand.finish_reason:
                candidate_info_str += f" 終了理由: {cand.finish_reason.name if hasattr(cand.finish_reason, 'name') else cand.finish_reason}"
            else:
                candidate_info_str += " 終了理由: N/A"

            if hasattr(cand, 'safety_ratings') and cand.safety_ratings:
                ratings_str = ", ".join([f"{sr.category.name if hasattr(sr.category, 'name') else sr.category}: {sr.probability.name if hasattr(sr.probability, 'name') else sr.probability}" for sr in cand.safety_ratings])
                candidate_info_str += f" 安全性評価: [{ratings_str}]"
            else:
                candidate_info_str += " 安全性評価: N/A"
    return candidate_info_str

# --- Gemini API Call and Popup Logic (変更なし) ---
def call_gemini_api_st(prompt_text, popup_type, popup_id, popup_title, related_links=None):
    model = get_gemini_model_st()
    if not model:
        st.session_state.active_gemini_popup = {"type": popup_type, "id": popup_id, "title": popup_title}
        st.session_state.gemini_popup_content = "エラー: Geminiモデルが利用できません。APIキーが設定されていないか、初期化に失敗しました。"
        st.session_state.gemini_popup_status = "モデルエラー"
        st.session_state.search_in_progress = False
        st.rerun()
        return

    st.session_state.active_gemini_popup = {"type": popup_type, "id": popup_id, "title": popup_title}
    st.session_state.gemini_popup_content = ""
    st.session_state.gemini_popup_status = "Gemini応答生成中... しばらくお待ちください"

    try:
        with st.spinner(f"Geminiに問い合わせ中: {popup_title}..."):
            response = model.generate_content(prompt_text, request_options={'timeout': 120})

        if response.prompt_feedback and response.prompt_feedback.block_reason:
            reason_name = response.prompt_feedback.block_reason.name if hasattr(response.prompt_feedback.block_reason, 'name') else str(response.prompt_feedback.block_reason)
            error_msg = f"Geminiからの応答がブロックされました。\n理由: {reason_name}"
            candidate_info_str = get_candidate_info(response)
            error_msg += candidate_info_str
            st.session_state.gemini_popup_content = error_msg
            st.session_state.gemini_popup_status = "エラー: 応答ブロック"
            st.error(error_msg)
        elif response.parts:
            result_text = "".join(part.text for part in response.parts if hasattr(part, 'text'))
            if not result_text:
                content_to_display = "(Geminiからの応答テキストが空です)"
                candidate_info_str = get_candidate_info(response)
                if candidate_info_str: content_to_display += f"\n{candidate_info_str}"
                st.session_state.gemini_popup_content = content_to_display
                st.session_state.gemini_popup_status = "Gemini応答表示完了 (空の応答)"
            else:
                result_text_md = re.sub(r"\[URL:\s*(https?://[^\s\]]+)\s*\]", r"[\1](\1)", result_text)
                content_to_display = result_text_md
                if related_links:
                    content_to_display += "\n\n---\n**関連リンク:**\n"
                    for idx, link_url in enumerate(related_links):
                        content_to_display += f"{idx+1}. [{link_url}]({link_url})\n"
                st.session_state.gemini_popup_content = content_to_display
                st.session_state.gemini_popup_status = "Gemini応答表示完了"
        else:
            st.session_state.gemini_popup_content = "(Geminiからの応答が予期せず空でした。Partsもありませんでした。)"
            candidate_info_str = get_candidate_info(response)
            if candidate_info_str: st.session_state.gemini_popup_content += f"\n{candidate_info_str}"
            st.session_state.gemini_popup_status = "エラー: 予期せぬ空応答"
            st.error(st.session_state.gemini_popup_content)

    except Exception as e:
        st.session_state.gemini_popup_content = f"Gemini処理中にエラーが発生しました: {type(e).__name__} - {e}"
        if 'response' in locals() and response:
             if hasattr(response, 'prompt_feedback') and response.prompt_feedback and response.prompt_feedback.block_reason:
                 reason_name = response.prompt_feedback.block_reason.name if hasattr(response.prompt_feedback.block_reason, 'name') else str(response.prompt_feedback.block_reason)
                 st.session_state.gemini_popup_content += f"\n(応答ブロック理由可能性: {reason_name})"
             candidate_info_str = get_candidate_info(response)
             if candidate_info_str: st.session_state.gemini_popup_content += candidate_info_str

        st.session_state.gemini_popup_status = "エラー: API呼び出し/処理失敗"
        st.error(st.session_state.gemini_popup_content)
    finally:
        st.session_state.search_in_progress = False
        st.rerun()

# --- Handler Functions for Actions (変更なし) ---
def handle_meeting_summary(meeting_url, meeting_name):
    if not GEMINI_API_KEY:
        st.toast("Gemini APIキー未設定のため、会議要約をスキップします。", icon="ℹ️")
        return
    st.session_state.search_in_progress = True
    processor = st.session_state.current_display_processor
    meeting_text_with_urls, speech_links = processor.get_meeting_details(meeting_url)
    if not meeting_text_with_urls:
        st.info("この会議の要約対象となる発言データが見つかりません。")
        st.session_state.search_in_progress = False
        st.rerun()
        return
    truncated_text, truncated = meeting_text_with_urls, False
    if len(meeting_text_with_urls) > GEMINI_MAX_INPUT_CHARS:
        truncated_text = meeting_text_with_urls[:GEMINI_MAX_INPUT_CHARS] + "\n... [文字数制限により以下省略]"
        truncated = True
        st.caption(f"注意: 会議 '{meeting_name}' の要約用コンテキストが {GEMINI_MAX_INPUT_CHARS} 文字で打ち切られました。")
    popup_title = f"会議要約: {meeting_name}" + (" (参照コンテキスト制限あり)" if truncated else "")
    prompt = f"""以下の日本の国会会議録の内容（各発言に `[URL: ...]` 形式で発言URLが付与されています）を、客観的に、主要な議題、議論のポイント、決定事項（もしあれば）がわかるように、箇条書き形式で要約してください。
    要約を作成する際には、以下の点を守ってください:
    1. 要約文中で言及した内容の根拠となった発言のURL `[URL: ...]` を、該当箇所の直後に引用として含めてください。
    2. 複数の発言を根拠とする場合は、それぞれのURLを示してください。
    # 会議名: {meeting_name}
    # 発言内容全文 (URL含む, 一部省略可能性あり):
    ---
    {truncated_text}
    ---
    # 要約 (上記指示に従い、根拠URL [URL: ...] を含めてください。箇条書き推奨):"""
    call_gemini_api_st(prompt, "summary", meeting_url, popup_title, related_links=speech_links)

def handle_speaker_analysis(speaker_name):
    if not GEMINI_API_KEY:
        st.toast("Gemini APIキー未設定のため、発言者分析をスキップします。", icon="ℹ️")
        return
    st.session_state.search_in_progress = True
    search_utterance_keyword = st.session_state.utterance_val.strip()
    if not search_utterance_keyword:
        st.info("分析の対象となる「発言」の検索キーワードが入力されていません。")
        st.session_state.search_in_progress = False
        st.rerun()
        return
    processor = st.session_state.current_display_processor
    speaker_text_with_urls, _ = processor.get_speaker_details(speaker_name)
    if not speaker_text_with_urls:
        st.info(f"{speaker_name} 氏の今回の検索結果における分析対象の発言データが見つかりません。")
        st.session_state.search_in_progress = False
        st.rerun()
        return
    truncated_text, truncated = speaker_text_with_urls, False
    if len(speaker_text_with_urls) > GEMINI_MAX_INPUT_CHARS:
         truncated_text = speaker_text_with_urls[:GEMINI_MAX_INPUT_CHARS] + "\n... [文字数制限により以下省略]"
         truncated = True
         st.caption(f"注意: 発言者 '{speaker_name}' の分析用コンテキストが {GEMINI_MAX_INPUT_CHARS} 文字で打ち切られました。")
    popup_title = f"発言者分析: {speaker_name} ({search_utterance_keyword} について)" + (" (参照コンテキスト制限あり)" if truncated else "")
    prompt = f"""以下の発言は、日本の国会議員である {speaker_name} 氏のものです。
    これらの発言内容全体（各発言に `[URL: ...]` 形式で発言URLが付与されています）を分析し、「{search_utterance_keyword}」というテーマに関して、同氏がどのような意見、主張、立場、提案をしているかを具体的に抽出・要約してください。
    分析結果を作成する際には、以下の点を守ってください:
    1. 分析結果の根拠となる具体的な発言箇所を特定してください。
    2. 特定した発言の `[URL: ...]` を、分析文中の該当箇所の直後に必ず引用として含めてください。複数の発言を根拠とする場合は、それぞれのURLを示してください。
    3. 客観的な視点で、箇条書きなどで分かりやすくまとめてください。
    # 発言者: {speaker_name}
    # 分析対象テーマ: {search_utterance_keyword}
    # 発言内容 (URL含む, 一部省略可能性あり):
    ---
    {truncated_text}
    ---
    # 「{search_utterance_keyword}」に関する {speaker_name} 氏の考え・主張の分析結果 (上記指示に従い、根拠URL [URL: ...] を必ず含めてください。箇条書き推奨):"""
    call_gemini_api_st(prompt, "analysis", speaker_name + "_" + search_utterance_keyword, popup_title)

def handle_speaker_links(speaker_name):
    processor = st.session_state.current_display_processor
    _ , speech_details = processor.get_speaker_details(speaker_name)
    if not speech_details:
         st.info(f"{speaker_name} 氏の発言は今回の検索結果内には見つかりませんでした。")
         return
    popup_title = f"発言リンク一覧: {speaker_name}"
    content = f"**{speaker_name} 氏の発言 ({len(speech_details)}件):**\n\n"
    for idx, detail in enumerate(speech_details):
        content += f"{idx+1}. {detail['context']} \n   "
        link_text = detail['snippet']
        speech_url = detail.get('url')
        if speech_url: content += f"[{link_text}]({speech_url})\n\n"
        else: content += f"{link_text} (URLなし)\n\n"
    st.session_state.active_gemini_popup = {"type": "speaker_links", "id": speaker_name, "title": popup_title}
    st.session_state.gemini_popup_content = content
    st.session_state.gemini_popup_status = "表示完了"
    st.rerun()

def start_ai_search_streamlit():
    if not GEMINI_API_KEY:
         st.warning("Gemini APIキーが設定されていません。AI検索機能は利用できません。")
         return
    st.session_state.search_in_progress = True
    question = st.session_state.ai_question_streamlit.strip()
    if not question:
        st.info("AIへの質問を入力してください。")
        st.session_state.search_in_progress = False
        st.rerun()
        return
    processor = st.session_state.current_display_processor
    if not processor or not processor.meeting_order:
        st.info("AIが参照する検索結果がありません。")
        st.session_state.search_in_progress = False
        st.rerun()
        return
    st.session_state.status_message = "AI検索のためのコンテキストを準備中..."
    try:
        context_text, truncated = processor.get_all_context_text()
        if not context_text:
             st.error("AI検索のコンテキスト生成に失敗しました。")
             st.session_state.search_in_progress = False
             st.rerun()
             return
        prompt = f"""以下の日本の国会会議録の検索結果（現在表示中のデータ全体）に基づいて、下記のユーザーの質問に回答してください。
        検索結果には、各発言に `[発言者: 名前 | 発言順: 番号 | URL: 発言ページのURL]` の形式で情報が付与されています。
        回答を作成する際には、以下の点を守ってください:
        1. 回答の根拠となる具体的な発言箇所を特定してください。
        2. 特定した発言の `[URL: ...]` を、回答文中の該当箇所の直後に必ず引用として含めてください。複数の発言を根拠とする場合は、それぞれのURLを示してください。
        3. 質問内容に対して、検索結果の範囲内で、客観的かつ正確に答えてください。
        # 検索結果コンテキスト:
        ---
        {context_text}
        ---
        # ユーザーの質問:
        {question}
        # 回答 (上記指示に従い、根拠となる発言URL [URL: ...] を必ず含めてください):"""
        popup_title = f"AI検索結果: {question[:30]}..." + (" (参照コンテキスト制限あり)" if truncated else "")
        call_gemini_api_st(prompt, "ai_search", question, popup_title)
    except ValueError as e:
         st.error(f"AI検索準備エラー: {e}")
         st.session_state.search_in_progress = False
         st.rerun()
    except Exception as e:
         st.error(f"AI検索準備中に予期せぬエラー: {e}")
         st.session_state.search_in_progress = False
         st.rerun()


# --- Main App ---
def main():
    st.set_page_config(page_title="国会会議録検索 Gemini拡張版", layout="wide")
    initialize_session_state()

    st.title("国会会議録検索 Gemini拡張版 v3 - Streamlit")

    if gemini_init_error: st.warning(gemini_init_error)

    with st.sidebar:
        st.header("検索条件")
        st.session_state.start_date_val = st.date_input("開始日", value=st.session_state.start_date_val,
                                                        min_value=st.session_state.min_date_limit,
                                                        max_value=st.session_state.max_date_limit,
                                                        disabled=st.session_state.search_in_progress)
        st.session_state.end_date_val = st.date_input("終了日", value=st.session_state.end_date_val,
                                                      min_value=st.session_state.min_date_limit,
                                                      max_value=st.session_state.max_date_limit,
                                                      disabled=st.session_state.search_in_progress)

        latest_session = datetime.date.today().year - 1947 + 155
        session_opts = ["-"] + [f"第{i}回国会" for i in range(latest_session, 0, -1)]
        st.session_state.session_val = st.selectbox("国会回次", session_opts, index=session_opts.index(st.session_state.session_val), disabled=st.session_state.search_in_progress)
        house_opts = ["-", "衆議院", "参議院", "両院"]
        st.session_state.house_val = st.selectbox("院名", house_opts, index=house_opts.index(st.session_state.house_val), disabled=st.session_state.search_in_progress)
        meeting_opts = ["-", "本会議", "委員会等"]
        st.session_state.meeting_type_val = st.selectbox("会議種別", meeting_opts, index=meeting_opts.index(st.session_state.meeting_type_val), disabled=st.session_state.search_in_progress)
        st.session_state.committee_val = st.text_input("委員会名", value=st.session_state.committee_val, disabled=st.session_state.search_in_progress)
        st.session_state.speaker_val = st.text_input("発言者", value=st.session_state.speaker_val, disabled=st.session_state.search_in_progress)
        st.session_state.utterance_val = st.text_input("発言（キーワード）", value=st.session_state.utterance_val, disabled=st.session_state.search_in_progress, help="ここに入力されたキーワードは、下の発言者別グラフの集計対象にもなります。")

        c1, c2 = st.columns(2)
        c1.button("検索実行", on_click=start_search_streamlit, type="primary", use_container_width=True, disabled=st.session_state.search_in_progress)
        c2.button("入力クリア", on_click=clear_search_fields_streamlit, use_container_width=True, disabled=st.session_state.search_in_progress)


    st.subheader("検索結果")
    if st.session_state.status_message: st.caption(st.session_state.status_message)

    proc = st.session_state.current_display_processor
    if proc and proc.meeting_order:
        total_pages_val = proc.total_pages
        if total_pages_val > 1:
            nav_cols = st.columns([1,1,3,1,1])
            if nav_cols[0].button("<< 前へ", disabled=(st.session_state.current_page <= 1 or st.session_state.search_in_progress), use_container_width=True):
                st.session_state.current_page -= 1
                st.session_state.page_select_key = str(st.session_state.current_page)
                st.session_state.page_select_key_bottom = str(st.session_state.current_page)
                st.rerun()
            if nav_cols[1].button("次へ >>", disabled=(st.session_state.current_page >= total_pages_val or st.session_state.search_in_progress), use_container_width=True):
                st.session_state.current_page += 1
                st.session_state.page_select_key = str(st.session_state.current_page)
                st.session_state.page_select_key_bottom = str(st.session_state.current_page)
                st.rerun()

            page_opts_list = [str(i) for i in range(1, total_pages_val + 1)]
            def page_change_callback(): # メイン結果のページネーション用
                st.session_state.current_page = int(st.session_state.page_select_key)
                st.session_state.page_select_key_bottom = st.session_state.page_select_key

            current_page_str = str(st.session_state.current_page)
            if current_page_str not in page_opts_list and page_opts_list:
                 current_page_str = page_opts_list[0]
                 st.session_state.current_page = int(current_page_str)
                 st.session_state.page_select_key = current_page_str
                 st.session_state.page_select_key_bottom = current_page_str


            nav_cols[2].selectbox("ページ:", options=page_opts_list, key="page_select_key", index=page_opts_list.index(current_page_str) if current_page_str in page_opts_list else 0, on_change=page_change_callback, disabled=st.session_state.search_in_progress, label_visibility="collapsed")
            nav_cols[4].write(f"{st.session_state.current_page} / {total_pages_val} ページ")

        display_page_streamlit(st.session_state.current_page)

        if total_pages_val > 1:
            st.divider()
            nav_cols_b = st.columns([1,1,3,1,1])
            if nav_cols_b[0].button("<< 前へ ", key="prev_b", disabled=(st.session_state.current_page <= 1 or st.session_state.search_in_progress), use_container_width=True):
                st.session_state.current_page -= 1
                st.session_state.page_select_key_bottom = str(st.session_state.current_page)
                st.session_state.page_select_key = str(st.session_state.current_page)
                st.rerun()
            if nav_cols_b[1].button("次へ >> ", key="next_b", disabled=(st.session_state.current_page >= total_pages_val or st.session_state.search_in_progress), use_container_width=True):
                st.session_state.current_page += 1
                st.session_state.page_select_key_bottom = str(st.session_state.current_page)
                st.session_state.page_select_key = str(st.session_state.current_page)
                st.rerun()

            def page_change_callback_bottom(): # メイン結果のページネーション用
                st.session_state.current_page = int(st.session_state.page_select_key_bottom)
                st.session_state.page_select_key = st.session_state.page_select_key_bottom

            current_page_str_b = str(st.session_state.current_page)
            if current_page_str_b not in page_opts_list and page_opts_list:
                 current_page_str_b = page_opts_list[0]
                 st.session_state.page_select_key_bottom = current_page_str_b
                 st.session_state.page_select_key = current_page_str_b

            nav_cols_b[2].selectbox("ページ: ", options=page_opts_list, key="page_select_key_bottom", index=page_opts_list.index(current_page_str_b) if current_page_str_b in page_opts_list else 0, on_change=page_change_callback_bottom, disabled=st.session_state.search_in_progress, label_visibility="collapsed")
            nav_cols_b[4].write(f"{st.session_state.current_page} / {total_pages_val} ページ")

        # 発言者別発言数グラフの表示
        if not st.session_state.speaker_speech_counts_df.empty:
            st.divider()
            st.subheader(f"「{st.session_state.utterance_val.strip()}」を含む発言の発言者別件数 (多い順)")

            df_for_graph = st.session_state.speaker_speech_counts_df.copy()
            # データは常に多い順で st.session_state に保存されている想定
            # df_for_graph = df_for_graph.sort_values('発言数', ascending=False) # 再確認

            st.bar_chart(df_for_graph.set_index('発言者')['発言数'], height=400)


            st.markdown("---")
            st.write("**発言者ごとのアクション:**")

            total_speakers = len(df_for_graph)
            total_speaker_pages = math.ceil(total_speakers / SPEAKERS_PER_GRAPH_PAGE)
            current_graph_speaker_page = st.session_state.graph_speaker_page

            if total_speaker_pages > 1:
                spk_nav_cols = st.columns([1, 1, 2, 1]) # 修正: プルダウン用に列を調整
                if spk_nav_cols[0].button("<< 前の議員", key="prev_spk_page_btn", disabled=(current_graph_speaker_page <= 1 or st.session_state.search_in_progress)):
                    st.session_state.graph_speaker_page -= 1
                    st.session_state.graph_speaker_page_select_key = str(st.session_state.graph_speaker_page) # selectboxも更新
                    st.rerun()

                speaker_page_opts = [str(i) for i in range(1, total_speaker_pages + 1)]
                def graph_speaker_page_change_callback():
                    st.session_state.graph_speaker_page = int(st.session_state.graph_speaker_page_select_key)
                    # st.rerun() は selectbox の on_change で暗黙的に行われることが多いが、明示しても良い

                current_graph_speaker_page_str = str(st.session_state.graph_speaker_page)
                if current_graph_speaker_page_str not in speaker_page_opts and speaker_page_opts:
                    current_graph_speaker_page_str = speaker_page_opts[0]
                    st.session_state.graph_speaker_page = int(current_graph_speaker_page_str)
                    st.session_state.graph_speaker_page_select_key = current_graph_speaker_page_str


                spk_nav_cols[1].selectbox(
                    "議員リストページ:",
                    options=speaker_page_opts,
                    key="graph_speaker_page_select_key",
                    index=speaker_page_opts.index(current_graph_speaker_page_str) if current_graph_speaker_page_str in speaker_page_opts else 0,
                    on_change=graph_speaker_page_change_callback,
                    disabled=st.session_state.search_in_progress,
                    label_visibility="collapsed"
                )

                if spk_nav_cols[2].button("次の議員 >>", key="next_spk_page_btn", disabled=(current_graph_speaker_page >= total_speaker_pages or st.session_state.search_in_progress)):
                    st.session_state.graph_speaker_page += 1
                    st.session_state.graph_speaker_page_select_key = str(st.session_state.graph_speaker_page) # selectboxも更新
                    st.rerun()
                spk_nav_cols[3].write(f"{current_graph_speaker_page} / {total_speaker_pages} ページ")


            start_idx = (current_graph_speaker_page - 1) * SPEAKERS_PER_GRAPH_PAGE
            end_idx = start_idx + SPEAKERS_PER_GRAPH_PAGE
            speakers_on_current_page = df_for_graph.iloc[start_idx:end_idx]


            for index, row in speakers_on_current_page.iterrows():
                speaker_name_for_btn = row['発言者']
                speech_count_for_btn = row['発言数']

                cols = st.columns([3, 1, 1])
                cols[0].markdown(f"**{speaker_name_for_btn}** (「{st.session_state.utterance_val.strip()}」関連発言: {speech_count_for_btn}件)")

                if cols[1].button("全発言一覧", key=f"spk_list_btn_{speaker_name_for_btn.replace(' ','_')}_{current_graph_speaker_page}", help=f"{speaker_name_for_btn}氏の全発言一覧を表示", use_container_width=True, disabled=st.session_state.search_in_progress):
                    handle_speaker_links(speaker_name_for_btn)

                can_analyze = GEMINI_API_KEY and st.session_state.utterance_val.strip()
                if cols[2].button("発言分析", key=f"spk_analyze_btn_{speaker_name_for_btn.replace(' ','_')}_{current_graph_speaker_page}", help=f"{speaker_name_for_btn}氏の「{st.session_state.utterance_val.strip()}」に関する発言を分析", disabled=not can_analyze or st.session_state.search_in_progress, use_container_width=True):
                    handle_speaker_analysis(speaker_name_for_btn)


    elif not st.session_state.search_in_progress:
        st.info("検索を実行するか、条件を変更して再度検索してください。")

    if proc and proc.meeting_order and not st.session_state.search_in_progress :
        with st.expander("詳細検索 (現在の検索結果に対して実行)", expanded=True):
            st.text_input("結果内検索（キーワードで絞り込み）:", key="filter_keyword_streamlit", value=st.session_state.filter_keyword_streamlit, disabled=st.session_state.search_in_progress)
            f_cols = st.columns(2)
            f_cols[0].button("絞り込み実行", on_click=filter_results_streamlit, use_container_width=True, disabled=st.session_state.search_in_progress)
            f_cols[1].button("全件表示に戻す", on_click=reset_filter_streamlit, use_container_width=True, disabled=(not st.session_state.is_filtered_mode or st.session_state.search_in_progress))
            if GEMINI_API_KEY:
                st.text_area("AIで質問 (現在表示中の結果について):", height=100, key="ai_question_streamlit", value=st.session_state.ai_question_streamlit, disabled=st.session_state.search_in_progress)
                st.button("質問する (Gemini)", on_click=start_ai_search_streamlit, disabled=st.session_state.search_in_progress)
            else:
                st.info("Gemini APIキーが設定されていないため、AI検索は利用できません。")

    if st.session_state.active_gemini_popup:
        popup_info = st.session_state.active_gemini_popup
        if hasattr(st, 'dialog'):
            @st.dialog(title=popup_info["title"])
            def show_dialog_content():
                st.markdown(st.session_state.gemini_popup_content, unsafe_allow_html=True)
                st.caption(st.session_state.gemini_popup_status)
                if st.button("閉じる", key=f"close_dialog_{popup_info['type']}_{popup_info['id']}"):
                    st.session_state.active_gemini_popup = None
                    st.rerun()
            show_dialog_content()
        else:
            with st.expander(f"Gemini結果: {popup_info['title']}", expanded=True):
                st.markdown(st.session_state.gemini_popup_content, unsafe_allow_html=True)
                st.caption(st.session_state.gemini_popup_status)
                if st.button("この結果を閉じる", key=f"close_exp_gemini_{popup_info['type']}_{popup_info['id']}"):
                    st.session_state.active_gemini_popup = None
                    st.rerun()

if __name__ == "__main__":
    main()
