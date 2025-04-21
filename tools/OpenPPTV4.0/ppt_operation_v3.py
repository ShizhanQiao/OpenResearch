import shutup

shutup.please()
import os
import uuid
import shutil
import zipfile
import xml.etree.ElementTree as ET
import re



def merge_pptx(ppt_a_dir, ppt_b_dir):
    """
    将PPT A的第一页幻灯片合并到PPT B中，并返回新的合并后的PPT文件路径。
    PPT A只有一页，PPT B可以有一页或多页。

    Args:
        ppt_a_dir (str): PPT A的文件夹路径（只有一页幻灯片）
        ppt_b_dir (str): PPT B的文件夹路径（一页或多页幻灯片）

    Returns:
        str: 合并后的新PPT文件路径（其实就是ppt b的路径）
    """

    # 命名空间定义
    content_type_ns = {'ns0': 'http://schemas.openxmlformats.org/package/2006/content-types'}
    app_ns = {'ns0': 'http://schemas.openxmlformats.org/officeDocument/2006/extended-properties',
              'vt': 'http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes'}
    ns_pres = {'p': 'http://schemas.openxmlformats.org/presentationml/2006/main',
               'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships',
               'a': 'http://schemas.openxmlformats.org/drawingml/2006/main'}
    rel_ns = {'': 'http://schemas.openxmlformats.org/package/2006/relationships'}
    ns_rels = {"r": "http://schemas.openxmlformats.org/package/2006/relationships"}


    # 确定PPT B中的幻灯片数量
    slides_dir_b = os.path.join(ppt_b_dir, "ppt", "slides")
    slide_files_b = [f for f in os.listdir(slides_dir_b) if f.startswith("slide") and f.endswith(".xml")]
    next_slide_num = len(slide_files_b) + 1
    new_slide_file = f"slide{next_slide_num}.xml"

    # 复制PPT A的slide1.xml到PPT B的新位置
    slide_a_path = os.path.join(ppt_a_dir, "ppt", "slides", "slide1.xml")
    slide_b_new_path = os.path.join(slides_dir_b, new_slide_file)
    shutil.copy2(slide_a_path, slide_b_new_path)

    # 复制PPT A的slide1.xml.rels到PPT B的新位置
    slides_rels_dir_a = os.path.join(ppt_a_dir, "ppt", "slides", "_rels")
    slides_rels_dir_b = os.path.join(slides_dir_b, "_rels")
    if not os.path.exists(slides_rels_dir_b):
        os.makedirs(slides_rels_dir_b)

    slide_rels_a_path = os.path.join(slides_rels_dir_a, "slide1.xml.rels")
    new_slide_rels_file = f"{new_slide_file}.rels"
    slide_rels_b_new_path = os.path.join(slides_rels_dir_b, new_slide_rels_file)

    # 先复制关系文件
    shutil.copy2(slide_rels_a_path, slide_rels_b_new_path)

    # 处理PPT A中的媒体文件
    media_dir_a = os.path.join(ppt_a_dir, "ppt", "media")
    media_dir_b = os.path.join(ppt_b_dir, "ppt", "media")

    if not os.path.exists(media_dir_b):
        os.makedirs(media_dir_b)

    # 媒体文件映射字典 {原文件名: 新文件名}
    media_mapping = {}

    # 如果PPT A有媒体文件夹且存在文件
    if os.path.exists(media_dir_a) and os.listdir(media_dir_a):
        for media_file in os.listdir(media_dir_a):
            media_file_path = os.path.join(media_dir_a, media_file)
            if os.path.isfile(media_file_path):
                # 生成新的文件名(使用UUID避免冲突)
                file_ext = os.path.splitext(media_file)[1]
                new_media_name = f"image_{uuid.uuid4().hex}{file_ext}"
                media_mapping[media_file] = new_media_name

                # 复制到PPT B的media文件夹
                new_media_path = os.path.join(media_dir_b, new_media_name)
                shutil.copy2(media_file_path, new_media_path)

    # 更新新幻灯片的关系文件中的媒体引用
    if media_mapping:
        ET.register_namespace('', rel_ns[''])

        rels_tree = ET.parse(slide_rels_b_new_path)
        rels_root = rels_tree.getroot()

        for rel in rels_root.findall("r:Relationship", ns_rels):
            target = rel.attrib.get("Target", "")
            if "../media/" in target:
                media_file = os.path.basename(target)
                if media_file in media_mapping:
                    rel.attrib["Target"] = f"../media/{media_mapping[media_file]}"

        rels_tree.write(slide_rels_b_new_path, encoding='utf-8', xml_declaration=True)

    # 更新[Content_Types].xml
    content_types_path = os.path.join(ppt_b_dir, "[Content_Types].xml")
    try:
        # 注册命名空间
        ET.register_namespace('', content_type_ns['ns0'])

        content_types_tree = ET.parse(content_types_path)
        content_types_root = content_types_tree.getroot()

        # 查找所有幻灯片的Override元素
        slide_overrides = []
        max_slide_num = 0

        for override in content_types_root.findall('{{{0}}}Override'.format(content_type_ns['ns0'])):
            part_name = override.attrib.get("PartName", "")
            if "/ppt/slides/slide" in part_name and part_name.endswith(".xml"):
                slide_overrides.append(override)
                pattern = r"/ppt/slides/slide(\d+)\.xml"
                match = re.search(pattern, part_name)
                if match:
                    slide_num = int(match.group(1))
                    max_slide_num = max(max_slide_num, slide_num)

        # 创建新的Override元素
        namespace = content_type_ns['ns0']
        new_override = ET.Element(f"{{{namespace}}}Override")
        new_override.attrib["PartName"] = f"/ppt/slides/{new_slide_file}"
        new_override.attrib["ContentType"] = "application/vnd.openxmlformats-officedocument.presentationml.slide+xml"

        # 在所有幻灯片Override的最后一个之后插入新元素
        if slide_overrides:
            last_slide_override = slide_overrides[-1]
            index = list(content_types_root).index(last_slide_override) + 1
            content_types_root.insert(index, new_override)
        else:
            content_types_root.append(new_override)

        content_types_tree.write(content_types_path, encoding='utf-8', xml_declaration=True)
    except Exception as e:
        raise Exception(f"更新Content_Types文件失败: {e}")

    # 更新docProps/app.xml中的幻灯片计数
    app_xml_path = os.path.join(ppt_b_dir, "docProps", "app.xml")
    try:
        # 注册命名空间
        ET.register_namespace('', app_ns['ns0'])
        ET.register_namespace('vt', app_ns['vt'])

        app_tree = ET.parse(app_xml_path)
        app_root = app_tree.getroot()

        # 更新Slides元素的计数
        slides_count_elem = app_root.find(".//ns0:Slides", app_ns)
        if slides_count_elem is not None:
            current_count = int(slides_count_elem.text)
            slides_count_elem.text = str(current_count + 1)

        # 更新TitlesOfParts
        TitlesOfParts_elem = app_root.find(".//ns0:TitlesOfParts", app_ns)
        if TitlesOfParts_elem is not None:
            vector_elem = TitlesOfParts_elem.find(".//vt:vector", app_ns)
            if vector_elem is not None:
                # 创建新的元素
                new_lpstr = ET.Element(f"{{{app_ns['vt']}}}lpstr")
                new_lpstr.text = "PowerPoint 演示文稿"

                # 添加到vector中
                vector_elem.append(new_lpstr)

                # 更新属性
                current_size = int(vector_elem.get("size", "0"))
                vector_elem.set("size", str(current_size + 1))

        # 更新HeadingPairs中的幻灯片计数
        heading_pairs_elem = app_root.find(".//ns0:HeadingPairs", app_ns)
        if heading_pairs_elem is not None:
            vector_elem = heading_pairs_elem.find(".//vt:vector", app_ns)
            if vector_elem is not None:
                variants = vector_elem.findall(".//vt:variant", app_ns)
                slide_title_index = None

                for i, variant in enumerate(variants):
                    lpstr = variant.find(".//vt:lpstr", app_ns)
                    if lpstr is not None and lpstr.text == "幻灯片标题":
                        slide_title_index = i
                        break

                if slide_title_index is not None and slide_title_index + 1 < len(variants):
                    count_variant = variants[slide_title_index + 1]
                    i4_elem = count_variant.find(".//vt:i4", app_ns)
                    if i4_elem is not None:
                        current_count = int(i4_elem.text)
                        i4_elem.text = str(current_count + 1)

        app_tree.write(app_xml_path, encoding='utf-8', xml_declaration=True)
    except Exception as e:
        pass

    # 更新presentation.xml中的幻灯片引用
    presentation_path = os.path.join(ppt_b_dir, "ppt", "presentation.xml")
    try:
        # 注册命名空间
        ET.register_namespace('p', ns_pres['p'])
        ET.register_namespace('r', ns_pres['r'])
        ET.register_namespace('a', ns_pres['a'])

        pres_tree = ET.parse(presentation_path)
        pres_root = pres_tree.getroot()

        # 查找sldIdLst元素
        sld_id_lst = pres_root.find(".//p:sldIdLst", ns_pres)
        if sld_id_lst is None:
            raise Exception("在presentation.xml中未找到sldIdLst元素")

        # 获取最大的幻灯片ID
        max_id = 255  # 默认起始ID
        for sld_id in sld_id_lst.findall(".//p:sldId", ns_pres):
            id_val = int(sld_id.attrib.get("id", "0"))
            if id_val > max_id:
                max_id = id_val

        # 更新presentation.xml.rels文件，确定新幻灯片的rId
        pres_rels_path = os.path.join(ppt_b_dir, "ppt", "_rels", "presentation.xml.rels")

        # 注册rels文件的命名空间
        ET.register_namespace('', 'http://schemas.openxmlformats.org/package/2006/relationships')

        pres_rels_tree = ET.parse(pres_rels_path)
        pres_rels_root = pres_rels_tree.getroot()

        # 寻找指向幻灯片的关系，确定插入位置和新的rId
        slide_relations = []
        for rel in pres_rels_root.findall(".//Relationship", {}):
            rid = rel.attrib.get("Id", "")
            rel_type = rel.attrib.get("Type", "")
            target = rel.attrib.get("Target", "")

            if "slide" in target and target.endswith(".xml") and rel_type.endswith("/slide"):
                slide_num = int(target.replace("slides/slide", "").replace(".xml", ""))
                slide_relations.append((rel, rid, slide_num))

        # 按幻灯片编号排序
        slide_relations.sort(key=lambda x: x[2])

        # 确定一个可用的rId编号
        max_rid_num = 0
        for rel in pres_rels_root:
            rid = rel.attrib.get("Id", "")
            if rid.startswith("rId"):
                try:
                    rid_num = int(rid[3:])
                    if rid_num > max_rid_num:
                        max_rid_num = rid_num
                except ValueError:
                    continue

        # 新幻灯片的rId
        new_rid = f"rId{max_rid_num + 1}"

        # 创建新的幻灯片关系元素
        new_rel = ET.Element("Relationship")
        new_rel.attrib["Id"] = new_rid
        new_rel.attrib["Type"] = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide"
        new_rel.attrib["Target"] = f"slides/{new_slide_file}"

        # 创建新的幻灯片引用
        p_namespace = ns_pres['p']
        r_namespace = ns_pres['r']
        new_slide_id = ET.Element(f"{{{p_namespace}}}sldId")
        new_slide_id.attrib["id"] = str(max_id + 1)
        new_slide_id.attrib[f"{{{r_namespace}}}id"] = new_rid

        # 将新的幻灯片引用添加到presentation.xml
        sld_id_lst.append(new_slide_id)
        pres_tree.write(presentation_path, encoding='utf-8', xml_declaration=True)

        # 重组presentation.xml.rels
        all_relations = []
        for rel in pres_rels_root:
            rid = rel.attrib.get("Id", "")
            rel_type = rel.attrib.get("Type", "")
            target = rel.attrib.get("Target", "")
            all_relations.append((rel, rid, rel_type, target))

        # 清除现有关系
        for rel in list(pres_rels_root):
            pres_rels_root.remove(rel)

        # 保留第一张幻灯片和母版的rId不变，为其他关系重新分配ID
        inserted = False
        for rel, rid, rel_type, target in all_relations:
            # 保持slide1和slideMaster的ID不变
            if ("slide1.xml" in target) or (rel_type.endswith("/slideMaster")):
                pres_rels_root.append(rel)
                # 在第一张幻灯片后插入新幻灯片
                if "slide1.xml" in target and not inserted:
                    pres_rels_root.append(new_rel)
                    inserted = True
            # 对于需要ID+1的关系
            elif rid.startswith("rId") and int(rid[3:]) >= int(new_rid[3:]):
                old_num = int(rid[3:])
                adjusted_rid = f"rId{old_num + 1}"
                rel.attrib["Id"] = adjusted_rid
                pres_rels_root.append(rel)
            # 其他关系保持不变
            else:
                pres_rels_root.append(rel)

        # 如果没有插入，添加到末尾
        if not inserted:
            pres_rels_root.append(new_rel)

        # 写回更新后的rels文件
        pres_rels_tree.write(pres_rels_path, encoding='utf-8', xml_declaration=True)
    except Exception as e:
        raise Exception(f"更新幻灯片引用关系失败: {e}")

    return ppt_b_dir





# 一些辅助函数
def unzip_pptx(pptx_path, output_path=None):
    """
    解压 pptx 文件到指定目录，
    如果未指定 output_path，则默认解压到当前工作目录下，
    并以 PPTX 文件名作为子目录。
    返回解压后的目录路径。
    """
    if output_path is None:
        output_path = os.getcwd()
    base_name = os.path.splitext(os.path.basename(pptx_path))[0]
    extract_dir = os.path.join(output_path, base_name)
    if not os.path.exists(extract_dir):
        os.makedirs(extract_dir)
    with zipfile.ZipFile(pptx_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    return extract_dir



def zip_dir(directory, output_file):
    """
    将指定目录及其所有子文件夹、文件打包（压缩）为一个 ZIP 文件，
    output_file 为输出的文件名。
    """
    with zipfile.ZipFile(output_file, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for foldername, subfolders, filenames in os.walk(directory):
            for filename in filenames:
                file_path = os.path.join(foldername, filename)
                # 保持相对路径结构（相对于目录根目录）
                arcname = os.path.relpath(file_path, directory)
                zip_file.write(file_path, arcname)

