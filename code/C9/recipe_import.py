"""
C9 Recipe Neo4j Importer — 将 AI Agent 解析的 recipe concepts/relationships 直连写入 Neo4j

使用 MERGE 语句保证幂等性，重复导入同一菜谱不会产生重复节点。
"""

import json
import logging
from typing import List, Dict, Any, Tuple

from neo4j import GraphDatabase

logger = logging.getLogger("recipe_import")


class RecipeNeo4jImporter:
    """将 parsed concepts + relationships 导入到运行中的 Neo4j 数据库"""

    @staticmethod
    def import_recipe_data(
        uri: str,
        user: str,
        password: str,
        concepts: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]],
        database: str = "neo4j",
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        导入 concepts 和 relationships 到 Neo4j。

        Args:
            uri: Neo4j Bolt URI
            user: 用户名
            password: 密码
            concepts: agent 输出的 concept dict 列表
            relationships: agent 输出的 relationship dict 列表
            database: Neo4j 数据库名

        Returns:
            (new_recipe_ids, new_ingredient_ids, new_step_ids) — 实际新创建的节点 ID
        """
        driver = GraphDatabase.driver(uri, auth=(user, password))

        new_recipe_ids = []
        new_ingredient_ids = []
        new_step_ids = []

        logger.info(f"开始 Neo4j 导入: {len(concepts)} concepts, {len(relationships)} relationships")
        try:
            with driver.session(database=database) as session:
                # 先写入所有 concept 节点
                created_count = 0
                matched_count = 0
                for concept in concepts:
                    is_new = RecipeNeo4jImporter._merge_concept(session, concept)
                    ctype = concept.get("concept_type", "")
                    cid = concept.get("concept_id", "")
                    name = concept.get("name", "")
                    if is_new:
                        created_count += 1
                        logger.debug(f"  [NEW] {ctype}: {name} (id={cid})")
                        if ctype == "Recipe":
                            new_recipe_ids.append(cid)
                        elif ctype == "Ingredient":
                            new_ingredient_ids.append(cid)
                        elif ctype == "CookingStep":
                            new_step_ids.append(cid)
                    else:
                        matched_count += 1
                        logger.debug(f"  [MATCH] {ctype}: {name} (id={cid})")

                logger.info(f"  concepts 写入完成: {created_count} new, {matched_count} matched")

                # 再写入所有关系
                rel_ok = 0
                rel_fail = 0
                for rel in relationships:
                    ok = RecipeNeo4jImporter._merge_relationship(session, rel)
                    if ok:
                        rel_ok += 1
                    else:
                        rel_fail += 1
                logger.info(f"  relationships 写入完成: {rel_ok} ok, {rel_fail} skipped")

        finally:
            driver.close()

        logger.info(
            f"Neo4j 导入完成: recipes={len(new_recipe_ids)} (新), "
            f"ingredients={len(new_ingredient_ids)} (新), "
            f"steps={len(new_step_ids)} (新)"
        )
        return new_recipe_ids, new_ingredient_ids, new_step_ids

    @staticmethod
    def _sanitize_prop(value):
        """将非原始类型的属性值转为 JSON 字符串，确保 Neo4j 兼容"""
        if value is None:
            return ""
        if isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, (list, tuple)):
            # 检查列表中是否全是原始类型
            if all(isinstance(x, (bool, int, float, str)) for x in value):
                return [str(x) if not isinstance(x, (bool, int, float)) else x for x in value]
            # 包含嵌套对象的列表 → JSON 字符串
            return json.dumps(value, ensure_ascii=False)
        # 字典或其他 → JSON 字符串
        return json.dumps(value, ensure_ascii=False)

    @staticmethod
    def _merge_concept(session, concept: Dict[str, Any]) -> bool:
        """MERGE 单个概念节点，返回 True 表示新建"""
        cid = concept["concept_id"]
        ctype = concept["concept_type"]
        name = concept.get("name", "")
        preferred_term = RecipeNeo4jImporter._sanitize_prop(concept.get("preferred_term", name))
        fsn = RecipeNeo4jImporter._sanitize_prop(concept.get("fsn", ""))
        synonyms = RecipeNeo4jImporter._sanitize_prop(concept.get("synonyms", ""))
        description = RecipeNeo4jImporter._sanitize_prop(concept.get("description", ""))

        if ctype == "Recipe":
            result = session.run(
                """
                MERGE (n:Recipe {nodeId: $cid})
                ON CREATE SET
                    n.name = $name, n.preferredTerm = $preferred_term,
                    n.fsn = $fsn, n.synonyms = $synonyms,
                    n.category = $category, n.difficulty = $difficulty,
                    n.cuisineType = $cuisine_type, n.prepTime = $prep_time,
                    n.cookTime = $cook_time, n.servings = $servings,
                    n.tags = $tags, n.filePath = $file_path
                ON MATCH SET
                    n.name = $name, n.preferredTerm = $preferred_term,
                    n.category = $category, n.difficulty = $difficulty,
                    n.cuisineType = $cuisine_type, n.prepTime = $prep_time,
                    n.cookTime = $cook_time, n.servings = $servings,
                    n.tags = $tags, n.filePath = $file_path
                RETURN n
                """,
                cid=cid,
                name=name,
                preferred_term=preferred_term,
                fsn=fsn,
                synonyms=synonyms,
                category=concept.get("category", ""),
                difficulty=concept.get("difficulty", 0),
                cuisine_type=concept.get("cuisine_type", ""),
                prep_time=concept.get("prep_time", ""),
                cook_time=concept.get("cook_time", ""),
                servings=concept.get("servings", ""),
                tags=concept.get("tags", ""),
                file_path=concept.get("file_path", ""),
            )
            # 检查是否真的创建了新节点（通过 ON CREATE 标志）
            # Neo4j 不直接返回此信息，我们用 EXISTS 检查前的状态
            summary = result.consume()
            # counters.nodes_created 在 MERGE 中同时包含 created 和 matched
            return summary.counters.nodes_created > 0

        elif ctype == "Ingredient":
            # 食材用 name 做 merge key，同名食材复用节点
            result = session.run(
                """
                MERGE (n:Ingredient {nodeId: $cid})
                ON CREATE SET
                    n.name = $name, n.preferredTerm = $preferred_term,
                    n.fsn = $fsn, n.synonyms = $synonyms,
                    n.category = $category, n.amount = $amount,
                    n.unit = $unit, n.isMain = $is_main
                ON MATCH SET
                    n.name = $name, n.category = $category,
                    n.amount = $amount, n.unit = $unit, n.isMain = $is_main
                RETURN n
                """,
                cid=cid,
                name=name,
                preferred_term=preferred_term,
                fsn=fsn,
                synonyms=synonyms,
                category=concept.get("category", ""),
                amount=concept.get("amount", ""),
                unit=concept.get("unit", ""),
                is_main=concept.get("is_main", False),
            )
            summary = result.consume()
            return summary.counters.nodes_created > 0

        elif ctype == "CookingStep":
            result = session.run(
                """
                MERGE (n:CookingStep {nodeId: $cid})
                ON CREATE SET
                    n.name = $name, n.preferredTerm = $preferred_term,
                    n.fsn = $fsn, n.description = $description,
                    n.stepNumber = $step_number, n.methods = $methods,
                    n.tools = $tools, n.timeEstimate = $time_estimate
                ON MATCH SET
                    n.name = $name, n.description = $description,
                    n.stepNumber = $step_number, n.methods = $methods,
                    n.tools = $tools, n.timeEstimate = $time_estimate
                RETURN n
                """,
                cid=cid,
                name=name,
                preferred_term=preferred_term,
                fsn=fsn,
                description=description,
                step_number=concept.get("step_number", 0),
                methods=concept.get("methods", ""),
                tools=concept.get("tools", ""),
                time_estimate=concept.get("time_estimate", ""),
            )
            summary = result.consume()
            return summary.counters.nodes_created > 0

        else:
            # 其他类型（Category, Root 等）：通用 MERGE
            result = session.run(
                f"""
                MERGE (n:{ctype} {{nodeId: $cid}})
                ON CREATE SET n.name = $name
                RETURN n
                """,
                cid=cid,
                name=name,
            )
            summary = result.consume()
            return summary.counters.nodes_created > 0

    @staticmethod
    def _merge_relationship(session, rel: Dict[str, Any]) -> bool:
        """MERGE 关系，返回 True 表示成功写入"""
        # 解析关系类型 ID 为可读字符串
        rt = rel.get("relationship_type", "")
        # 常见类型映射
        type_map = {
            "801000001": "REQUIRES",
            "801000002": "REQUIRES_TOOL",
            "801000003": "CONTAINS_STEP",
            "801000004": "BELONGS_TO_CATEGORY",
            "801000005": "HAS_DIFFICULTY",
            "801000006": "USES_METHOD",
            "801000007": "HAS_AMOUNT",
            "801000008": "STEP_FOLLOWS",
            "801000009": "SERVES_PEOPLE",
            "801000010": "COOKING_TIME",
            "801000011": "PREP_TIME",
        }
        rel_type = type_map.get(str(rt), rt)

        source_id = rel.get("source_id", "")
        target_id = rel.get("target_id", "")

        if not source_id or not target_id:
            logger.debug(f"  [SKIP] relationship: missing source_id or target_id")
            return False

        # 属性处理
        props = {}
        for k, v in rel.items():
            if k not in ("relationship_id", "source_id", "target_id", "relationship_type"):
                props[k] = v

        try:
            if props:
                # 带属性的关系用 MERGE + SET
                prop_items = list(props.items())
                set_clause = ", ".join(f"r.{k} = ${k}" for k, _ in prop_items)
                params = {
                    "sid": source_id,
                    "tid": target_id,
                    "rel_type": rel_type,
                }
                for k, v in prop_items:
                    params[k] = v

                session.run(
                    f"""
                    MATCH (a {{nodeId: $sid}})
                    MATCH (b {{nodeId: $tid}})
                    MERGE (a)-[r:{rel_type}]->(b)
                    SET {set_clause}
                    """,
                    **params,
                )
            else:
                session.run(
                    f"""
                    MATCH (a {{nodeId: $sid}})
                    MATCH (b {{nodeId: $tid}})
                    MERGE (a)-[r:{rel_type}]->(b)
                    """,
                    sid=source_id,
                    tid=target_id,
                    rel_type=rel_type,
                )
            return True
        except Exception as e:
            logger.warning(f"  [FAIL] relationship {rel_type} ({source_id}->{target_id}): {e}")
            return False

    @staticmethod
    def recipe_exists(uri: str, user: str, password: str, recipe_name: str, database: str = "neo4j") -> bool:
        """检查同名菜谱是否已存在于 Neo4j"""
        driver = GraphDatabase.driver(uri, auth=(user, password))
        try:
            with driver.session(database=database) as session:
                result = session.run(
                    "MATCH (r:Recipe {name: $name}) RETURN count(r) as cnt",
                    name=recipe_name,
                )
                record = result.single()
                return record and record["cnt"] > 0
        finally:
            driver.close()

    @staticmethod
    def delete_recipe_cascade(uri: str, user: str, password: str, recipe_name: str, database: str = "neo4j") -> int:
        """
        级联删除菜谱及其独有的 Ingredient/CookingStep 节点和所有关系。
        共享的 Ingredient 节点不会被删除。
        返回删除的节点总数。
        """
        driver = GraphDatabase.driver(uri, auth=(user, password))
        try:
            with driver.session(database=database) as session:
                # 先删除 recipe 独有的关联节点（没有被其他 recipe 引用的）
                result = session.run(
                    """
                    MATCH (r:Recipe {name: $name})
                    // 先删除该 recipe 发出的所有关系
                    OPTIONAL MATCH (r)-[rel]->()
                    DELETE rel
                    WITH r
                    // 删除 CookingStep（每个 recipe 独有）
                    OPTIONAL MATCH (r)-[:CONTAINS_STEP]->(s:CookingStep)
                    DETACH DELETE s
                    WITH r
                    // 删除只被当前 recipe 引用的 Ingredient
                    OPTIONAL MATCH (r)-[:REQUIRES]->(i:Ingredient)
                    WITH r, i
                    OPTIONAL MATCH (other:Recipe)-[:REQUIRES]->(i)
                    WHERE other <> r
                    WITH r, i, count(other) AS other_count
                    WHERE other_count = 0
                    DETACH DELETE i
                    WITH r
                    // 最后删除 recipe 自身
                    DETACH DELETE r
                    """,
                    name=recipe_name,
                )
                record = result.single()
                logger.info(f"级联删除菜谱 '{recipe_name}': 完成")
                return 1 if record else 0
        finally:
            driver.close()
