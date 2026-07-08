"""
真正的图RAG检索模块
基于图结构的知识推理和检索，而非简单的关键词匹配
"""

import json
import logging
import time
from collections import defaultdict, deque
from typing import List, Dict, Tuple, Any, Optional, Set
from dataclasses import dataclass
from enum import Enum

from langchain_core.documents import Document
from neo4j import GraphDatabase

from logging_setup import get_dedicated_file_logger

# Text2Cypher 专用日志（只写文件，不进控制台/主日志）
t2c_log = get_dedicated_file_logger("text2cypher", "text2cypher")

logger = logging.getLogger(__name__)

class QueryType(Enum):
    """查询类型枚举"""
    ENTITY_RELATION = "entity_relation"  # 实体关系查询：A和B有什么关系？
    MULTI_HOP = "multi_hop"  # 多跳查询：A通过什么连接到C？
    SUBGRAPH = "subgraph"  # 子图查询：A相关的所有信息
    PATH_FINDING = "path_finding"  # 路径查找：从A到B的最佳路径
    CLUSTERING = "clustering"  # 聚类查询：和A相似的都有什么？

@dataclass
class GraphQuery:
    """图查询结构"""
    query_type: QueryType
    source_entities: List[str]
    target_entities: List[str] = None
    relation_types: List[str] = None
    max_depth: int = 2
    max_nodes: int = 50
    constraints: Dict[str, Any] = None

@dataclass
class GraphPath:
    """图路径结构"""
    nodes: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]
    path_length: int
    relevance_score: float
    path_type: str

@dataclass
class KnowledgeSubgraph:
    """知识子图结构"""
    central_nodes: List[Dict[str, Any]]
    connected_nodes: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]
    graph_metrics: Dict[str, float]
    reasoning_chains: List[List[str]]

class GraphRAGRetrieval:
    """
    真正的图RAG检索系统
    核心特点：
    1. 查询意图理解：识别图查询模式
    2. 多跳图遍历：深度关系探索
    3. 子图提取：相关知识网络
    4. 图结构推理：基于拓扑的推理
    5. 动态查询规划：自适应遍历策略
    """
    
    def __init__(self, config, llm_client):
        self.config = config
        self.llm_client = llm_client
        self.driver = None
        
        # 图结构缓存
        self.entity_cache = {}
        self.relation_cache = {}
        self.subgraph_cache = {}
        
    def initialize(self):
        """初始化图RAG检索系统"""
        logger.info("初始化图RAG检索系统...")
        
        # 连接Neo4j
        try:
            self.driver = GraphDatabase.driver(
                self.config.neo4j_uri, 
                auth=(self.config.neo4j_user, self.config.neo4j_password)
            )
            # 测试连接
            with self.driver.session() as session:
                session.run("RETURN 1")
            logger.info("Neo4j连接成功")
        except Exception as e:
            logger.error(f"Neo4j连接失败: {e}")
            return
        
        # 预热：构建实体和关系索引
        self._build_graph_index()
        
    def _build_graph_index(self):
        """构建图索引以加速查询"""
        logger.info("构建图结构索引...")
        
        try:
            with self.driver.session() as session:
                # 构建实体索引 - 修复Neo4j语法兼容性问题
                entity_query = """
                MATCH (n)
                WHERE n.nodeId IS NOT NULL
                WITH n, COUNT { (n)--() } as degree
                RETURN labels(n) as node_labels, n.nodeId as node_id, 
                       n.name as name, n.category as category, degree
                ORDER BY degree DESC
                LIMIT 1000
                """
                
                result = session.run(entity_query)
                for record in result:
                    node_id = record["node_id"]
                    self.entity_cache[node_id] = {
                        "labels": record["node_labels"],
                        "name": record["name"],
                        "category": record["category"],
                        "degree": record["degree"]
                    }
                
                # 构建关系类型索引
                relation_query = """
                MATCH ()-[r]->()
                RETURN type(r) as rel_type, count(r) as frequency
                ORDER BY frequency DESC
                """
                
                result = session.run(relation_query)
                for record in result:
                    rel_type = record["rel_type"]
                    self.relation_cache[rel_type] = record["frequency"]
                    
                logger.info(f"索引构建完成: {len(self.entity_cache)}个实体, {len(self.relation_cache)}个关系类型")
                
        except Exception as e:
            logger.error(f"构建图索引失败: {e}")
    
    def understand_graph_query(self, query: str) -> GraphQuery:
        """
        理解查询的图结构意图
        这是图RAG的核心：从自然语言到图查询的转换
        """
        prompt = f"""
        作为图数据库专家，分析以下查询的图结构意图，并将自然语言问题映射到**已有图结构**上。
        
        已知图中大致有以下节点和关系：
        - 节点类型：
          - Recipe：菜谱节点，包含 name、description、cuisineType（如"川菜"）、category、tags、prepTime、cookTime 等属性
          - Ingredient：食材节点，包含 name、category（如"蔬菜"、"蛋白质" 等）
          - Category：菜品分类（如"川菜"、"家常菜"、"素菜"）
          - CookingStep：烹饪步骤
        - 主要关系：
          - (Recipe)-[:REQUIRES]->(Ingredient)
          - (Recipe)-[:BELONGS_TO_CATEGORY]->(Category)
          - (Recipe)-[:CONTAINS_STEP]->(CookingStep)
        
        请根据上述图结构分析下面的查询：
        
        查询：{query}
        
        请识别：
       
        1. source_entities：
           - 只包含在图中**很有可能有对应节点**的具体实体名称
           - 优先选择：菜系（如"川菜"）、具体菜名（如"宫保鸡丁"）、食材名（如"鸡肉"、"豆腐"）
           - 不要把抽象概念或约束（如"糖尿病饮食限制"、"具体川菜菜品"、"健康饮食"、"30分钟内"）放进 source_entities
        
        2. target_entities：
           - 只在确实需要限制「路径终点」时填写
           - 同样只能使用可能出现在 Recipe / Ingredient / Category 节点上的名称（如"蔬菜"、"素菜"、具体菜名）
           - 如果不确定目标实体怎么映射到图中，请返回空列表 []
        
        3. relation_types：本次推理中希望优先考虑的关系类型列表
           - 例如：["REQUIRES", "BELONGS_TO_CATEGORY"]
        
        4. max_depth：建议的图遍历深度（1-3 之间的整数）
        
        5. constraints：可选的**属性级约束**，用于表达图结构之外的过滤条件，例如：
           - 健康/饮食限制（如"糖尿病"、"低糖"）
           - 时间限制（如"30分钟内"）
           - 口味偏好（如"清淡"、"少油"）
           用一个字典描述，例如：
           {{
             "health": ["糖尿病", "低糖"],
             "time": {{"max_minutes": 30}},
             "style": ["川菜"]
           }}
        
        示例1：
        查询："鸡肉配什么蔬菜好？"
        期望分析：这是 multi_hop 查询，需要通过"鸡肉→使用鸡肉的菜品→这些菜品使用的蔬菜"的路径推理。
        
        返回JSON示例：
        {{
          "query_type": "multi_hop",
          "source_entities": ["鸡肉"],
          "target_entities": ["蔬菜"],
          "relation_types": ["REQUIRES", "BELONGS_TO_CATEGORY"],
          "max_depth": 3,
          "constraints": {{}}
        }}
        
        示例2：
        查询："适合糖尿病人吃的低糖川菜有哪些，并且制作时间不超过30分钟？"
        期望分析：
          - 图中可以直接对应的实体：主要是菜系 "川菜"
          - 糖尿病/低糖/30分钟 属于属性级约束，不能当作节点
          - 可以使用 subgraph 或 multi_hop，以 "川菜" 为核心实体，结合属性约束做后续过滤
        
        返回JSON示例：
        {{
          "query_type": "subgraph",
          "source_entities": ["川菜"],
          "target_entities": [],
          "relation_types": ["BELONGS_TO_CATEGORY", "REQUIRES"],
          "max_depth": 2,
          "constraints": {{
            "health": ["糖尿病", "低糖"],
            "time": {{"max_minutes": 30}}
          }}
        }}
        
        请严格返回一个合法的 JSON 对象，不要包含任何多余的说明文字。
        """
        
        try:
            response = self.llm_client.chat.completions.create(
                model=self.config.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1000
            )
            
            result = json.loads(response.choices[0].message.content.strip())
               
            return GraphQuery(
                query_type=QueryType(result.get("query_type", "subgraph")),
                source_entities=result.get("source_entities", []),
                target_entities=result.get("target_entities", []),
                relation_types=result.get("relation_types", []),
                max_depth=result.get("max_depth", 2),
                max_nodes=50
            )
            
        except Exception as e:
            logger.error(f"查询意图理解失败: {e}")
            # 降级方案：默认子图查询
            return GraphQuery(
                query_type=QueryType.SUBGRAPH,
                source_entities=[query],
                max_depth=2
            )
    
    def multi_hop_traversal(self, graph_query: GraphQuery) -> List[GraphPath]:
        """
        多跳图遍历：这是图RAG的核心优势
        通过图结构发现隐含的知识关联
        """
        logger.info(f"执行多跳遍历: {graph_query.source_entities} -> {graph_query.target_entities}")
        
        paths = []
        
        if not self.driver:
            logger.error("Neo4j连接未建立")
            return paths
            
        try:
            with self.driver.session() as session:
                # 构建多跳遍历查询
                source_entities = graph_query.source_entities
                target_keywords = graph_query.target_entities or []
                max_depth = graph_query.max_depth
                
                # 根据查询类型选择不同的遍历策略
                if graph_query.query_type in (QueryType.MULTI_HOP, QueryType.ENTITY_RELATION):
                    # 根据是否有目标关键词动态拼接过滤条件
                    target_filter_clause = ""
                    if target_keywords:
                        target_filter_clause = """
                    AND ANY(kw IN $target_keywords WHERE
                        (target.name IS NOT NULL AND (toString(target.name) CONTAINS kw OR kw CONTAINS toString(target.name))) OR
                        (target.category IS NOT NULL AND (toString(target.category) CONTAINS kw OR kw CONTAINS toString(target.category)))
                    )"""
                    
                    cypher_query = f"""
                    // 多跳推理查询
                    UNWIND $source_entities as source_name
                    MATCH (source)
                    WHERE source.name CONTAINS source_name OR source.nodeId = source_name
                    
                    // 执行多跳遍历
                    MATCH path = (source)-[*1..{max_depth}]-(target)
                    WHERE NOT source = target{target_filter_clause}
                    
                    // 计算路径相关性
                    WITH path, source, target,
                         length(path) as path_len,
                         relationships(path) as rels,
                         nodes(path) as path_nodes
                    
                    // 路径评分：短路径 + 高度数节点 + 关系类型匹配
                    WITH path, source, target, path_len, rels, path_nodes,
                         (1.0 / path_len) + 
                         (REDUCE(s = 0.0, n IN path_nodes | s + COUNT {{ (n)--() }}) / 10.0 / size(path_nodes)) +
                         (CASE WHEN ANY(r IN rels WHERE type(r) IN $relation_types) THEN 0.3 ELSE 0.0 END) as relevance
                    
                    ORDER BY relevance DESC
                    LIMIT 20
                    
                    RETURN path, source, target, path_len, rels, path_nodes, relevance
                    """
                    
                    params = {
                        "source_entities": source_entities,
                        "relation_types": graph_query.relation_types or []
                    }
                    if target_keywords:
                        params["target_keywords"] = target_keywords
                    
                    result = session.run(cypher_query, params)
                    
                    for record in result:
                        path_data = self._parse_neo4j_path(record)
                        if path_data:
                            paths.append(path_data)
                
                elif graph_query.query_type == QueryType.ENTITY_RELATION:
                    # 实体间关系查询
                    paths.extend(self._find_entity_relations(graph_query, session))
                
                elif graph_query.query_type == QueryType.PATH_FINDING:
                    # 最短路径查找
                    paths.extend(self._find_shortest_paths(graph_query, session))
                    
        except Exception as e:
            logger.error(f"多跳遍历失败: {e}")
            
        logger.info(f"多跳遍历完成，找到 {len(paths)} 条路径")
        return paths
    
    def extract_knowledge_subgraph(self, graph_query: GraphQuery) -> KnowledgeSubgraph:
        """
        提取知识子图：获取实体相关的完整知识网络
        这体现了图RAG的整体性思维
        """
        logger.info(f"提取知识子图: {graph_query.source_entities}")
        
        if not self.driver:
            logger.error("Neo4j连接未建立")
            return self._fallback_subgraph_extraction(graph_query)
        
        try:
            with self.driver.session() as session:
                # 简化的子图提取（不依赖APOC）
                cypher_query = f"""
                // 找到源实体
                UNWIND $source_entities as entity_name
                MATCH (source)
                WHERE source.name CONTAINS entity_name 
                   OR source.nodeId = entity_name
                
                // 获取指定深度的邻居
                MATCH (source)-[r*1..{graph_query.max_depth}]-(neighbor)
                WITH source, collect(DISTINCT neighbor) as neighbors, 
                     collect(DISTINCT r) as relationships
                WHERE size(neighbors) <= $max_nodes
                
                // 计算图指标
                WITH source, neighbors, relationships,
                     size(neighbors) as node_count,
                     size(relationships) as rel_count
                
                RETURN 
                    source,
                    neighbors[0..{graph_query.max_nodes}] as nodes,
                    relationships[0..{graph_query.max_nodes}] as rels,
                    {{
                        node_count: node_count,
                        relationship_count: rel_count,
                        density: CASE WHEN node_count > 1 THEN toFloat(rel_count) / (node_count * (node_count - 1) / 2) ELSE 0.0 END
                    }} as metrics
                """
                
                result = session.run(cypher_query, {
                    "source_entities": graph_query.source_entities,
                    "max_nodes": graph_query.max_nodes
                })
                
                record = result.single()
                if record:
                    return self._build_knowledge_subgraph(record)
                    
        except Exception as e:
            logger.error(f"子图提取失败: {e}")
            
        # 降级方案：简单邻居查询
        return self._fallback_subgraph_extraction(graph_query)
    
    def graph_structure_reasoning(self, subgraph: KnowledgeSubgraph, query: str) -> List[str]:
        """
        基于图结构的推理：这是图RAG的智能之处
        不仅检索信息，还能进行逻辑推理
        """
        reasoning_chains = []
        
        try:
            # 1. 识别推理模式
            reasoning_patterns = self._identify_reasoning_patterns(subgraph)
            
            # 2. 构建推理链
            for pattern in reasoning_patterns:
                chain = self._build_reasoning_chain(pattern, subgraph)
                if chain:
                    reasoning_chains.append(chain)
            
            # 3. 验证推理链的可信度
            validated_chains = self._validate_reasoning_chains(reasoning_chains, query)
            
            logger.info(f"图结构推理完成，生成 {len(validated_chains)} 条推理链")
            return validated_chains
            
        except Exception as e:
            logger.error(f"图结构推理失败: {e}")
            return []
    
    def adaptive_query_planning(self, query: str) -> List[GraphQuery]:
        """
        自适应查询规划：根据查询复杂度动态调整策略
        """
        # 分析查询复杂度
        complexity_score = self._analyze_query_complexity(query)
        
        query_plans = []
        
        if complexity_score < 0.3:
            # 简单查询：直接邻居查询
            plan = GraphQuery(
                query_type=QueryType.ENTITY_RELATION,
                source_entities=[query],
                max_depth=1,
                max_nodes=20
            )
            query_plans.append(plan)
            
        elif complexity_score < 0.7:
            # 中等复杂度：多跳查询
            plan = GraphQuery(
                query_type=QueryType.MULTI_HOP,
                source_entities=[query],
                max_depth=2,
                max_nodes=50
            )
            query_plans.append(plan)
            
        else:
            # 复杂查询：子图提取 + 推理
            plan1 = GraphQuery(
                query_type=QueryType.SUBGRAPH,
                source_entities=[query],
                max_depth=3,
                max_nodes=100
            )
            plan2 = GraphQuery(
                query_type=QueryType.MULTI_HOP,
                source_entities=[query],
                max_depth=3,
                max_nodes=50
            )
            query_plans.extend([plan1, plan2])
            
        return query_plans
    
    def graph_rag_search(self, query: str, top_k: int = 5) -> List[Document]:
        """
        图RAG主搜索接口：整合所有图RAG能力
        """
        logger.info(f"开始图RAG检索: {query}")
        
        if not self.driver:
            logger.warning("Neo4j连接未建立，返回空结果")
            return []
        
        # 1. 查询意图理解
        graph_query = self.understand_graph_query(query)
        logger.info(f"查询类型: {graph_query.query_type.value}")
        
        results = []
        
        try:
            # 2. 根据查询类型执行不同策略
            if graph_query.query_type in [QueryType.MULTI_HOP, QueryType.PATH_FINDING]:
                # 多跳遍历 / 路径查找
                paths = self.multi_hop_traversal(graph_query)
                results.extend(self._paths_to_documents(paths, query))
                
            elif graph_query.query_type in [QueryType.SUBGRAPH, QueryType.CLUSTERING]:
                # 子图提取 / 聚类查询：都视为“围绕核心实体的局部知识网络”
                subgraph = self.extract_knowledge_subgraph(graph_query)
                
                # 图结构推理
                reasoning_chains = self.graph_structure_reasoning(subgraph, query)
                
                results.extend(self._subgraph_to_documents(subgraph, reasoning_chains, query))
                
            elif graph_query.query_type == QueryType.ENTITY_RELATION:
                # 实体关系查询（可以视为一跳 / 少量跳的路径查询）
                paths = self.multi_hop_traversal(graph_query)
                results.extend(self._paths_to_documents(paths, query))
            
            # 3. 图结构相关性排序
            results = self._rank_by_graph_relevance(results, query)
            
            logger.info(f"图RAG检索完成，返回 {len(results[:top_k])} 个结果")
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"图RAG检索失败: {e}")
            return []

    def multi_hop_search(self, query: str, top_k: int = 5) -> List[Document]:
        """
        多跳检索（Text2Cypher 优先 + 代码模板兜底）

        适用场景：意图识别已确定为 multi-hop

        流程：
          1. Text2Cypher：让 LLM 直接生成 Cypher 并执行（主路径）
          2. 失败/无结果 → 降级到 understand_graph_query + 多跳模板（兜底）
        """
        logger.info(f"┌─[MULTI-HOP SEARCH] query='{query[:60]}'")
        t0 = time.time()

        if not self.driver:
            logger.warning("│ Neo4j未连接，跳过图检索")
            logger.info(f"└─[MULTI-HOP SEARCH] end ({(time.time()-t0)*1000:.1f}ms)")
            return []

        # ── 阶段 1: Text2Cypher（主路径） ──
        logger.info("│ [PHASE 1] Text2Cypher（LLM 直接生成 Cypher）...")
        docs = self._text2cypher_search(query, top_k)

        # ── 阶段 2: 模板兜底 ──
        if not docs:
            logger.info("│ [PHASE 2] Text2Cypher 无结果，降级到模板路径")
            docs = self._template_multi_hop_search(query, top_k)

        logger.info(f"│ 最终返回 {len(docs[:top_k])} 个文档 (总耗时 {(time.time()-t0)*1000:.1f}ms)")
        logger.info(f"└─[MULTI-HOP SEARCH] end")
        return docs[:top_k]

    # ─────────────────────────────────────────
    # Text2Cypher 主路径
    # ─────────────────────────────────────────

    def _text2cypher_search(self, query: str, top_k: int = 5) -> List[Document]:
        """
        Text2Cypher：LLM 直接根据自然语言生成 Cypher 并执行

        返回 [] 表示让调用方降级到模板路径

        详细产物（生成的 Cypher / 完整 records）写入专用日志文件：
          logs/text2cypher_YYYY-MM-DD.log
        """
        if not self.llm_client:
            logger.info("│  无 LLM client，跳过 Text2Cypher")
            return []

        # ── 专用日志：标记本次 Text2Cypher 调用的开始 ──
        t2c_log.info("")
        t2c_log.info("─" * 60)
        t2c_log.info(f"[TEXT2CYPHER CALL]  query = {query}")

        # 1. 获取实际 schema（避免 LLM 编造不存在的标签/关系）
        schema_summary = self._get_graph_schema_summary()
        logger.info(f"│  当前图 schema: {schema_summary.replace(chr(10), ' | ')}")
        t2c_log.info(f"[SCHEMA]  {schema_summary}")

        # 2. LLM 生成 Cypher
        t_llm = time.time()
        cypher = self._llm_generate_cypher(query, schema_summary)
        llm_ms = (time.time() - t_llm) * 1000
        logger.info(f"│  LLM 生成 Cypher 耗时 {llm_ms:.1f}ms")
        logger.info(f"│  生成 Cypher:\n│    {cypher}")
        t2c_log.info(f"[LLM]  生成耗时 {llm_ms:.1f}ms")
        t2c_log.info(f"[CYPHER]\n{cypher if cypher else '(空)'}")

        if not cypher:
            t2c_log.info("[RESULT] LLM 未生成 Cypher，降级")
            return []

        # 3. 安全检查：禁止写/删/索引操作
        if not self._is_safe_cypher(cypher):
            logger.warning("│  Cypher 未通过安全检查（包含写/危险操作）")
            t2c_log.info("[RESULT] 安全检查未通过，降级")
            return []

        # 4. 执行
        try:
            t_db = time.time()
            with self.driver.session() as session:
                result = session.run(cypher)
                records = [dict(r) for r in result]
            db_ms = (time.time() - t_db) * 1000
            logger.info(f"│  Neo4j 执行 {db_ms:.1f}ms, 返回 {len(records)} 条记录")
            t2c_log.info(f"[DB]  执行 {db_ms:.1f}ms, 返回 {len(records)} 条")
        except Exception as e:
            logger.warning(f"│  Text2Cypher 执行失败: {e}")
            t2c_log.info(f"[DB ERROR] {e}")
            return []

        # ── 专用日志：完整 records（json 格式） ──
        if records:
            t2c_log.info(f"[RECORDS]  {len(records)} 条记录（完整 dump）：")
            for i, rec in enumerate(records, 1):
                # 把不可序列化的值转成字符串
                safe_rec = {k: (str(v) if not isinstance(v, (str, int, float, bool, list, dict, type(None))) else v)
                            for k, v in rec.items()}
                t2c_log.info(f"  [{i}] {safe_rec}")
        else:
            t2c_log.info("[RECORDS]  0 条")

        if not records:
            logger.info("│  Text2Cypher 返回 0 条记录")
            return []

        # 5. 转 Document
        docs = self._records_to_documents(records, query, top_k)
        logger.info(f"│  Text2Cypher 命中 {len(docs)} 个文档")
        t2c_log.info(f"[DOCS]  生成 {len(docs)} 个 Document")
        t2c_log.info("─" * 60)
        return docs

    def _get_graph_schema_summary(self) -> str:
        """从 Neo4j 读取实际 schema（节点标签 + 关系类型），用于 prompt"""
        try:
            with self.driver.session() as session:
                labels = [r["label"] for r in session.run("CALL db.labels()")]
                rels = [r["rt"] for r in session.run("CALL db.relationshipTypes()")]
                label_str = ", ".join(sorted(labels)) if labels else "(无)"
                rel_str = ", ".join(sorted(rels)) if rels else "(无)"
                return f"节点标签: [{label_str}]  关系类型: [{rel_str}]"
        except Exception as e:
            logger.debug(f"读 schema 失败: {e}")
            return "节点标签: [Recipe, Ingredient, CookingStep, Category]  关系类型: [REQUIRES, BELONGS_TO_CATEGORY, CONTAINS_STEP, HAS_CONCEPT_TYPE]"

    def _llm_generate_cypher(self, query: str, schema_summary: str) -> Optional[str]:
        """调用 LLM 直接生成 Cypher

        关键点：必须同时兼容普通模型（content 即答案）和推理模型
        （如 deepseek-v4-flash / deepseek-reasoner：content 为空，
        答案在 reasoning_content，且会消耗 max_tokens 用于思考）。
        策略：
          1. 优先用非推理模型（config.t2c_llm_model，可独立配置）
          2. 如该模型就是推理模型，通过 extra_body={"thinking": {"type": "disabled"}}
             尝试关闭推理（DeepSeek 支持）
          3. 解析时同时读 content 和 reasoning_content（兜底）
        """
        prompt = (
            "你是 Neo4j Cypher 专家。根据用户的中文烹饪问题，生成一条只读的 Cypher 查询，"
            "用于从图数据库中检索相关菜谱。\n\n"
            f"当前图 schema:\n{schema_summary}\n\n"
            "===== 实际节点与字段（已验证可用） =====\n"
            "Recipe 节点: name, cuisineType(川菜/粤菜/湘菜/西北菜...), stepCount(整数), ingredientCount(整数), nodeId\n"
            "Ingredient 节点: name, category(调料/蔬菜/蛋白质/其他/淀粉类/脂肪/工具), isMain(true=主料/false=辅料), unit, amount\n"
            "CookingStep 节点: name(如'步骤1'), description(步骤正文), tools(列表), methods(列表)\n"
            "Category/RecipeCategory: 菜品大分类(素菜/荤菜/水产/早餐/主食/汤类/甜品/饮料)\n"
            "CookingMethod/CookingTool: 节点名都是'烹饪方法'/'工具'字面量，过滤用 BELONGS_TO 关系 + 看 CookingStep.tools/methods 列表更靠谱\n\n"
            "===== 关键关系（按使用频率） =====\n"
            "(Recipe)-[:REQUIRES]->(Ingredient)         ← 食材相关查询的主路径\n"
            "(Recipe)-[:CONTAINS_STEP]->(CookingStep)   ← 查步骤正文用 s.description\n"
            "(Recipe)-[:BELONGS_TO_CATEGORY]->(RecipeCategory)  ← 菜品类型（素菜/荤菜等）\n"
            "(Recipe)-[:BELONGS_TO]->(CookingMethod/CookingTool) ← 注意节点名几乎都是字面量\n"
            "(Recipe)-[:SIMILAR]->(Recipe)              ← 相似菜推荐\n"
            "(Recipe)-[:NEXT_STEP]->(CookingStep)       ← 步骤顺序（一般不用）\n\n"
            "===== 硬规则 =====\n"
            "1. 只用 MATCH / WHERE / RETURN / WITH / OPTIONAL MATCH / ORDER BY / LIMIT，禁止 CREATE/MERGE/DELETE/SET/DROP/CALL db.*\n"
            "2. 食材名匹配用 (i.name CONTAINS 'X' OR i.name = 'X')，中文食材经常部分匹配\n"
            "3. \"蔬菜/调料/蛋白质\" 等品类用 i.category = '蔬菜' 这种枚举值，**绝不要**用 CONTAINS 中文偏旁 OR 链拼凑\n"
            "4. 菜系用 r.cuisineType = '川菜'，菜品大类用 BELONGS_TO_CATEGORY\n"
            "5. \"主料是X的菜\" 一定要加 i.isMain = true（避免调料/水干扰）\n"
            "6. 步骤正文在 s.description，步骤序号在 s.name（不是 stepNumber）\n"
            "7. 必须 RETURN DISTINCT + LIMIT ≤ 30；至少返回 r.name AS recipe_name\n"
            "8. **时间约束**：Recipe 节点没有 cookTime/prepTime 字段！用户问\"30分钟内\"只能近似用 r.stepCount <= N 代替，并在 RETURN 里说明这是\"步骤数\"，不要试图按分钟硬过滤\n"
            "9. **健康/饮食限制**（糖尿病/低脂/低糖）数据中没有对应字段，只能用相近食材（比如\"瘦肉/鱼类/蔬菜\"）粗筛，并尽量搭配 r.ingredientCount 较小的菜\n\n"
            "===== Few-shot 示例（已用真实数据验证可执行） =====\n\n"
            "示例 1（食材搭配 + 类别）\n"
            "Q: 鸡肉搭配什么蔬菜比较好\n"
            "Cypher:\n"
            "MATCH (r:Recipe)-[:REQUIRES]->(chicken:Ingredient)\n"
            "WHERE chicken.name CONTAINS '鸡肉' OR chicken.name = '鸡肉'\n"
            "MATCH (r)-[:REQUIRES]->(veg:Ingredient)\n"
            "WHERE veg.category = '蔬菜'\n"
            "RETURN DISTINCT r.name AS recipe_name, veg.name AS vegetable_name\n"
            "LIMIT 10\n\n"
            "示例 2（菜系 + 步骤数过滤）\n"
            "Q: 川菜里有哪些简单的做法\n"
            "Cypher:\n"
            "MATCH (r:Recipe)\n"
            "WHERE r.cuisineType = '川菜' AND r.stepCount <= 8\n"
            "RETURN r.name AS recipe_name, r.stepCount AS steps\n"
            "ORDER BY r.stepCount\n"
            "LIMIT 10\n\n"
            "示例 3（主料查询，必须 isMain）\n"
            "Q: 土豆能做什么菜\n"
            "Cypher:\n"
            "MATCH (r:Recipe)-[rel:REQUIRES]->(i:Ingredient)\n"
            "WHERE i.isMain = true AND (i.name CONTAINS '土豆' OR i.name = '土豆')\n"
            "RETURN DISTINCT r.name AS recipe_name\n"
            "LIMIT 10\n\n"
            "示例 4（步骤正文，description 字段）\n"
            "Q: 红烧肉怎么做\n"
            "Cypher:\n"
            "MATCH (r:Recipe)-[:CONTAINS_STEP]->(s:CookingStep)\n"
            "WHERE r.name CONTAINS '红烧肉' AND s.description IS NOT NULL\n"
            "RETURN r.name AS recipe_name, s.name AS step_label, s.description AS step_desc\n"
            "ORDER BY r.name, s.name\n"
            "LIMIT 30\n\n"
            "示例 5（多食材笛卡尔积）\n"
            "Q: 牛肉和土豆能一起做什么菜\n"
            "Cypher:\n"
            "MATCH (r:Recipe)-[:REQUIRES]->(i1:Ingredient)\n"
            "WHERE i1.name CONTAINS '牛肉' OR i1.name = '牛肉'\n"
            "MATCH (r)-[:REQUIRES]->(i2:Ingredient)\n"
            "WHERE i2.name CONTAINS '土豆' OR i2.name = '土豆'\n"
            "RETURN DISTINCT r.name AS recipe_name\n"
            "LIMIT 10\n\n"
            "示例 6（相似菜，SIMILAR 关系）\n"
            "Q: 和红烧鱼相似的菜有哪些\n"
            "Cypher:\n"
            "MATCH (r:Recipe)-[:SIMILAR]->(s:Recipe)\n"
            "WHERE r.name CONTAINS '红烧'\n"
            "RETURN DISTINCT r.name AS base, s.name AS similar_dish\n"
            "LIMIT 10\n\n"
            "用户问题：" + query + "\n\n"
            "只输出一条 Cypher 语句，不要任何解释、Markdown 代码块、其他文字。"
        )

        # 选模型：Text2Cypher 专用模型 > 主模型
        model = getattr(self.config, "t2c_llm_model", None) or self.config.llm_model
        # 推理模型必须给足 token（思考+输出），普通模型给 400 就够
        is_reasoning_model = any(
            kw in model.lower() for kw in ("reasoner", "r1", "v4", "thinking", "o1", "o3")
        )
        max_tokens = 1500 if is_reasoning_model else 400

        # 如果是 DeepSeek 推理模型，尝试关闭思考（DeepSeek v3.2+ 支持）
        extra_body = None
        if "deepseek" in model.lower() and is_reasoning_model:
            extra_body = {"thinking": {"type": "disabled"}}

        try:
            kwargs = dict(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                
            )
            if extra_body:
                kwargs["extra_body"] = extra_body

            resp = self.llm_client.chat.completions.create(**kwargs)
            msg = resp.choices[0].message

            # ====== 关键：同时读取 content 和 reasoning_content ======
            content = (getattr(msg, "content", None) or "").strip()
            reasoning = (getattr(msg, "reasoning_content", None) or "").strip()

            # 记录到 Text2Cypher 专用日志，便于排查
            t2c_log.info(f"[RAW RESP] model={model}  reasoning_model={is_reasoning_model}")
            t2c_log.info(f"[RAW RESP] content_len={len(content)}  reasoning_len={len(reasoning)}")
            t2c_log.info(f"[RAW RESP] usage={resp.usage}")
            if reasoning:
                # 思考过程过长，只记前 300 字避免日志爆炸
                t2c_log.info(f"[REASONING] {reasoning[:300]}{'...' if len(reasoning) > 300 else ''}")

            # 优先用 content；如果 content 为空但 reasoning 有内容（推理模型行为），用 reasoning
            raw = content if content else reasoning
            if not raw:
                t2c_log.info("[RAW RESP] content 和 reasoning_content 都为空")
                return None

            # 去 Markdown 围栏
            text = raw.replace("```cypher", "").replace("```", "").strip()

            # 解析多行 Cypher（关键：不能只取第一行！）
            # 多行 Cypher 形如：
            #   MATCH (r)-[:R]->(i)
            #   WHERE i.name CONTAINS '鸡肉'
            #   RETURN r.name
            # 旧逻辑 "取首行" 会把它截断成 "MATCH (r)-[:R]->(i)"，
            # 送到 Neo4j 后报 "Query cannot conclude with MATCH"。
            #
            # 策略：
            #   1. 找 Cypher 起点（首个 MATCH / OPTIONAL MATCH / WITH / CALL / RETURN 等）
            #   2. 从起点起保留所有非空行
            #   3. 遇到连续两个空行认为进入下一条语句，截断（防止多段拼接）
            #   4. 不主动检测 "下一个 MATCH" 作为语句边界，因为合法 Cypher
            #      经常用连续 MATCH 形成笛卡尔积（如本例的 鸡肉 → 蔬菜 双 MATCH）
            _CYPHER_START = ("MATCH", "OPTIONAL", "WITH", "CALL", "RETURN", "UNWIND")
            lines = text.splitlines()
            start_idx = None
            for i, line in enumerate(lines):
                first_word = line.strip().split("(", 1)[0].split(" ", 1)[0].upper()
                if first_word in _CYPHER_START:
                    start_idx = i
                    break
            if start_idx is None:
                t2c_log.info(f"[PARSE] 找不到 Cypher 起始关键词，原文:\n{text[:500]}")
                return None

            # 从起点起拼接，遇到连续两个空行截断
            cypher_lines = []
            blank_streak = 0
            for line in lines[start_idx:]:
                if not line.strip():
                    blank_streak += 1
                    if blank_streak >= 2:
                        break  # 两条语句之间的空行
                    continue  # 单个空行保留（不强制）
                blank_streak = 0
                cypher_lines.append(line)

            cypher = "\n".join(cypher_lines).strip()
            cypher = cypher.rstrip(";").strip()
            t2c_log.info(f"[PARSE] 提取 Cypher ({len(cypher)} chars, {len(cypher_lines)} 行):\n{cypher}")
            return cypher if cypher else None
        except Exception as e:
            logger.warning(f"│  LLM 生成 Cypher 失败: {e}")
            t2c_log.info(f"[RAW RESP] 调用异常: {e}")
            return None

    # 写操作关键词（任何匹配即拒绝）
    _FORBIDDEN_CYPHER_KEYWORDS = (
        "CREATE", "MERGE", "DELETE", "DETACH", "SET ", "REMOVE",
        "DROP", "CALL db.", "LOAD CSV", "FOREACH",
    )

    def _is_safe_cypher(self, cypher: str) -> bool:
        """简单的 Cypher 安全检查：禁止写操作"""
        upper = cypher.upper()
        for kw in self._FORBIDDEN_CYPHER_KEYWORDS:
            if kw in upper:
                logger.warning(f"│  检测到禁止关键字: {kw}")
                return False
        return True

    def _records_to_documents(self, records: List[dict], query: str, top_k: int) -> List[Document]:
        """把 Neo4j 返回的 records 转成 Document 列表

        适配两种格式：
          A) records 是 Recipe 节点全字段（含 description / 食材 / 步骤）
          B) records 是 path nodes（带 labels）
        """
        documents = []
        for rec in records[:top_k]:
            # 兼容 Recipe 字段或 path 节点格式
            name = (rec.get("recipe_name") or rec.get("name")
                    or (rec.get("r") or {}).get("name") or "图检索结果")

            parts = [f"【菜谱】{name}"]
            if rec.get("category"):
                parts.append(f"分类：{rec['category']}")
            if rec.get("cuisine") or rec.get("cuisineType"):
                parts.append(f"菜系：{rec.get('cuisine') or rec.get('cuisineType')}")
            if rec.get("description"):
                parts.append(f"简介：{rec['description']}")
            if rec.get("ingredients"):
                ings = rec["ingredients"] if isinstance(rec["ingredients"], list) else [str(rec["ingredients"])]
                parts.append(f"食材：{', '.join(map(str, ings[:20]))}")
            if rec.get("steps"):
                steps = rec["steps"] if isinstance(rec["steps"], list) else [str(rec["steps"])]
                step_lines = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(steps[:20]))
                parts.append(f"步骤：\n{step_lines}")
            # 兜底：其他字段拼成 description
            extras = {k: v for k, v in rec.items()
                      if k not in {"recipe_name", "name", "category", "cuisine",
                                   "cuisineType", "description", "ingredients", "steps"}
                      and v is not None}
            if extras and len(parts) == 1:
                parts.append("其他字段：" + ", ".join(f"{k}={v}" for k, v in extras.items()))

            doc = Document(
                page_content="\n".join(parts),
                metadata={
                    "search_type": "text2cypher",
                    "recipe_name": name,
                    "raw_keys": list(rec.keys()),
                },
            )
            documents.append(doc)
        return documents

    # ─────────────────────────────────────────
    # 模板兜底路径（原 multi_hop_search 实现）
    # ─────────────────────────────────────────

    def _template_multi_hop_search(self, query: str, top_k: int = 5) -> List[Document]:
        """模板路径兜底：understand_graph_query → 多跳 Cypher 模板"""
        logger.info("│  [TEMPLATE PATH] understand_graph_query...")
        try:
            t_llm = time.time()
            graph_query = self.understand_graph_query(query)
            llm_ms = (time.time() - t_llm) * 1000
            logger.info(f"│  LLM 推断 GraphQuery 耗时 {llm_ms:.1f}ms")
            logger.info(f"│  query_type      = {graph_query.query_type.value}")
            logger.info(f"│  source_entities = {graph_query.source_entities}")
            logger.info(f"│  target_entities = {graph_query.target_entities}")
            logger.info(f"│  relation_types  = {graph_query.relation_types}")
            logger.info(f"│  max_depth       = {graph_query.max_depth}")

            if graph_query.query_type != QueryType.MULTI_HOP:
                logger.info(f"│  LLM 返回 {graph_query.query_type.value}，强制覆盖为 multi_hop")
                graph_query.query_type = QueryType.MULTI_HOP

            if not graph_query.source_entities:
                logger.warning("│  source_entities 为空，降级到 _quick_extract_entities")
                fallback = self._quick_extract_entities(query)
                logger.info(f"│  fallback entities = {fallback}")
                if not fallback:
                    return []
                graph_query.source_entities = fallback

            # 执行模板 Cypher
            logger.info("│  [TEMPLATE PATH] 执行多跳模板 Cypher...")
            t_db = time.time()
            paths = self.multi_hop_traversal(graph_query)
            db_ms = (time.time() - t_db) * 1000
            logger.info(f"│  Cypher {db_ms:.1f}ms, 找到 {len(paths)} 条路径")

            if paths:
                for i, p in enumerate(paths[:5], 1):
                    if hasattr(p, "nodes"):
                        node_names = [n.get("name", "?") for n in p.nodes[:5]]
                        logger.info(f"│    [{i}] len={p.path_length}  rel={p.relevance_score:.3f}  "
                                    f"nodes={'→'.join(node_names)}")

            docs = self._paths_to_documents(paths, query)
            logger.info(f"│  [TEMPLATE PATH] 返回 {len(docs[:top_k])} 个文档")
            return docs[:top_k]
        except Exception as e:
            logger.error(f"│  模板路径失败: {e}")
            return []

    def _quick_extract_entities(self, query: str) -> List[str]:
        """
        轻量实体提取（仅提取食材/菜品/菜系名）

        优先用 LLM（max_tokens=80, temperature=0），失败/超时降级到 jieba
        """
        # 1) LLM 路径
        if self.llm_client is not None:
            prompt = (
                "从以下烹饪查询中提取关键实体（食材名、菜名、菜系），用于 Neo4j 图谱检索。\n"
                "只返回 JSON 数组，例如 [\"鸡肉\", \"土豆\"]。\n"
                "如果查询中没有具体实体（例如只是抽象描述），返回 []。\n\n"
                f"查询：{query}\n\n"
                "只输出 JSON 数组，不要任何其他文字。"
            )
            try:
                resp = self.llm_client.chat.completions.create(
                    model=self.config.llm_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=80,
                )
                text = resp.choices[0].message.content.strip()
                import re as _re
                m = _re.search(r'\[.*?\]', text, _re.DOTALL)
                if m:
                    import json as _json
                    entities = _json.loads(m.group(0))
                    cleaned = [str(e).strip() for e in entities if e and str(e).strip()]
                    if cleaned:
                        return cleaned
            except Exception as e:
                logger.warning(f"LLM 实体提取失败，降级到 jieba: {e}")

        # 2) jieba 降级路径
        return self._jieba_extract_entities(query)

    @staticmethod
    def _jieba_extract_entities(query: str) -> List[str]:
        """jieba 分词 + 长度过滤作为 LLM 失败的降级方案"""
        try:
            import jieba
            tokens = [t.strip() for t in jieba.cut(query)
                      if 2 <= len(t.strip()) <= 6]
            # 简单去停用词
            stop = {"什么", "怎么", "如何", "为什么", "哪个", "哪些", "可以", "能做",
                    "推荐", "介绍", "做菜", "做一下", "能做", "菜吗", "一下", "什么菜"}
            tokens = [t for t in tokens if t not in stop]
            return tokens[:5]
        except ImportError:
            return []

    # ========== 辅助方法 ==========
    
    def _parse_neo4j_path(self, record) -> Optional[GraphPath]:
        """解析Neo4j路径记录"""
        try:
            path_nodes = []
            for node in record["path_nodes"]:
                path_nodes.append({
                    "id": node.get("nodeId", ""),
                    "name": node.get("name", ""),
                    "labels": list(node.labels),
                    "properties": dict(node)
                })
            
            relationships = []
            for rel in record["rels"]:
                relationships.append({
                    "type": type(rel).__name__,
                    "properties": dict(rel)
                })
            
            return GraphPath(
                nodes=path_nodes,
                relationships=relationships,
                path_length=record["path_len"],
                relevance_score=record["relevance"],
                path_type="multi_hop"
            )
            
        except Exception as e:
            logger.error(f"路径解析失败: {e}")
            return None
    
    def _build_knowledge_subgraph(self, record) -> KnowledgeSubgraph:
        """构建知识子图对象"""
        try:
            central_nodes = [dict(record["source"])]
            connected_nodes = [dict(node) for node in record["nodes"]]
            relationships = [dict(rel) for rel in record["rels"]]
            
            return KnowledgeSubgraph(
                central_nodes=central_nodes,
                connected_nodes=connected_nodes,
                relationships=relationships,
                graph_metrics=record["metrics"],
                reasoning_chains=[]
            )
        except Exception as e:
            logger.error(f"构建知识子图失败: {e}")
            return KnowledgeSubgraph(
                central_nodes=[],
                connected_nodes=[],
                relationships=[],
                graph_metrics={},
                reasoning_chains=[]
            )
    
    def _paths_to_documents(self, paths: List[GraphPath], query: str) -> List[Document]:
        """将图路径转换为Document对象"""
        documents = []

        for i, path in enumerate(paths):
            # 1. 路径结构（简短）
            path_desc = self._build_path_description(path)

            # 2. 路径中 Recipe 节点的完整内容（description / 食材 / 步骤）
            recipe_blocks = self._fetch_recipe_details(path)

            # 3. 组合 page_content：路径结构 + 菜谱正文
            if recipe_blocks:
                page_content = path_desc + "\n\n" + "\n\n---\n\n".join(recipe_blocks)
            else:
                page_content = path_desc

            doc = Document(
                page_content=page_content,
                metadata={
                    "search_type": "graph_path",
                    "path_length": path.path_length,
                    "relevance_score": path.relevance_score,
                    "path_type": path.path_type,
                    "node_count": len(path.nodes),
                    "relationship_count": len(path.relationships),
                    "recipe_name": path.nodes[0].get("name", "图结构结果") if path.nodes else "图结构结果",
                    "has_recipe_content": bool(recipe_blocks),
                }
            )
            documents.append(doc)

        return documents

    def _fetch_recipe_details(self, path: GraphPath) -> List[str]:
        """
        从 Neo4j 拉取路径中 Recipe 节点的详细内容（描述/食材/步骤）

        容错处理：
          - 多种关系类型（REQUIRES / CONTAINS_STEP / HAS_CONCEPT_TYPE）
          - 节点可能没有 nodeId，用 name 兜底
          - 失败返回 []（不阻断主流程）
        """
        if not self.driver:
            return []

        # 收集 Recipe 节点的标识
        recipe_keys = []  # [(id_or_name, label)]
        for n in path.nodes:
            labels = n.get("labels") or []
            if "Recipe" in labels:
                rid = n.get("id") or n.get("name")
                if rid:
                    recipe_keys.append(rid)

        if not recipe_keys:
            return []

        try:
            with self.driver.session() as session:
                # 用 nodeId 或 name 匹配；尽量拉 description + 食材 + 步骤
                cypher = """
                MATCH (r:Recipe)
                WHERE r.nodeId IN $keys OR r.name IN $keys
                OPTIONAL MATCH (r)-[rel1]->(s)
                  WHERE type(rel1) IN ['CONTAINS_STEP', 'HAS_STEP'] AND (s:CookingStep OR s:Step)
                OPTIONAL MATCH (r)-[rel2]->(i:Ingredient)
                WITH r,
                     collect(DISTINCT s.name) AS steps,
                     collect(DISTINCT i.name) AS ingredients
                RETURN r.name        AS name,
                       r.description AS description,
                       r.cuisineType AS cuisine,
                       r.category    AS category,
                       r.tags        AS tags,
                       steps,
                       ingredients
                """
                result = session.run(cypher, {"keys": recipe_keys})
                blocks = []
                for rec in result:
                    parts = [f"【菜谱】{rec['name']}"]
                    if rec["category"]:
                        parts.append(f"分类：{rec['category']}")
                    if rec["cuisine"]:
                        parts.append(f"菜系：{rec['cuisine']}")
                    if rec["tags"]:
                        # tags 可能是 list 或 string
                        tags = rec["tags"] if isinstance(rec["tags"], list) else [str(rec["tags"])]
                        parts.append(f"标签：{', '.join(tags)}")
                    if rec["description"]:
                        parts.append(f"简介：{rec['description']}")
                    if rec["ingredients"]:
                        parts.append(f"食材：{', '.join(rec['ingredients'][:30])}")
                    if rec["steps"]:
                        step_lines = "\n".join(
                            f"  {idx+1}. {s}" for idx, s in enumerate(rec["steps"][:30])
                        )
                        parts.append(f"步骤：\n{step_lines}")
                    blocks.append("\n".join(parts))
                if blocks:
                    logger.debug(f"  [RECIPE DETAILS] 拉取 {len(blocks)} 个菜谱详情")
                return blocks
        except Exception as e:
            logger.warning(f"拉取 Recipe 详情失败: {e}")
            return []
    
    def _subgraph_to_documents(self, subgraph: KnowledgeSubgraph, 
                              reasoning_chains: List[str], query: str) -> List[Document]:
        """将知识子图转换为Document对象"""
        documents = []
        
        # 子图整体描述
        subgraph_desc = self._build_subgraph_description(subgraph)
        
        doc = Document(
            page_content=subgraph_desc,
            metadata={
                "search_type": "knowledge_subgraph",
                "node_count": len(subgraph.connected_nodes),
                "relationship_count": len(subgraph.relationships),
                "graph_density": subgraph.graph_metrics.get("density", 0.0),
                "reasoning_chains": reasoning_chains,
                "recipe_name": subgraph.central_nodes[0].get("name", "知识子图") if subgraph.central_nodes else "知识子图"
            }
        )
        documents.append(doc)
        
        return documents
    
    def _build_path_description(self, path: GraphPath) -> str:
        """构建路径的自然语言描述"""
        if not path.nodes:
            return "空路径"

        desc_parts = []
        for i, node in enumerate(path.nodes):
            name = node.get("name", f"节点{i}")
            labels = node.get("labels") or []
            props = node.get("properties") or {}

            # Recipe 节点：补充简介（兜底，防止 _fetch_recipe_details 失败时也丢内容）
            if "Recipe" in labels and props.get("description"):
                name = f"{name}（{props['description'][:80]}）"
            desc_parts.append(name)

            if i < len(path.relationships):
                rel_type = path.relationships[i].get("type", "相关")
                desc_parts.append(f" --{rel_type}--> ")

        return "".join(desc_parts)
    
    def _build_subgraph_description(self, subgraph: KnowledgeSubgraph) -> str:
        """构建子图的自然语言描述"""
        central_names = [node.get("name", "未知") for node in subgraph.central_nodes]
        node_count = len(subgraph.connected_nodes)
        rel_count = len(subgraph.relationships)
        
        return f"关于 {', '.join(central_names)} 的知识网络，包含 {node_count} 个相关概念和 {rel_count} 个关系。"
    
    def _rank_by_graph_relevance(self, documents: List[Document], query: str) -> List[Document]:
        """基于图结构相关性排序"""
        return sorted(documents, 
                     key=lambda x: x.metadata.get("relevance_score", 0.0), 
                     reverse=True)
    
    def _analyze_query_complexity(self, query: str) -> float:
        """分析查询复杂度"""
        complexity_indicators = ["什么", "如何", "为什么", "哪些", "关系", "影响", "原因"]
        score = sum(1 for indicator in complexity_indicators if indicator in query)
        return min(score / len(complexity_indicators), 1.0)
    
    def _identify_reasoning_patterns(self, subgraph: KnowledgeSubgraph) -> List[str]:
        """识别推理模式"""
        return ["因果关系", "组成关系", "相似关系"]
    
    def _build_reasoning_chain(self, pattern: str, subgraph: KnowledgeSubgraph) -> Optional[str]:
        """构建推理链"""
        return f"基于{pattern}的推理链"
    
    def _validate_reasoning_chains(self, chains: List[str], query: str) -> List[str]:
        """验证推理链"""
        return chains[:3]
    
    def _find_entity_relations(self, graph_query: GraphQuery, session) -> List[GraphPath]:
        """查找实体间关系"""
        return []
    
    def _find_shortest_paths(self, graph_query: GraphQuery, session) -> List[GraphPath]:
        """查找最短路径"""
        return []
    
    def _fallback_subgraph_extraction(self, graph_query: GraphQuery) -> KnowledgeSubgraph:
        """降级子图提取"""
        return KnowledgeSubgraph(
            central_nodes=[],
            connected_nodes=[],
            relationships=[],
            graph_metrics={},
            reasoning_chains=[]
        )
    
    def refresh_cache(self):
        """清空并重建实体/关系缓存（增量更新后调用）"""
        self.entity_cache.clear()
        self.relation_cache.clear()
        self._build_graph_index()
        logger.info(f"GraphRAG 缓存已刷新: {len(self.entity_cache)} 实体, {len(self.relation_cache)} 关系类型")

    def close(self):
        """关闭资源连接"""
        if hasattr(self, 'driver') and self.driver:
            self.driver.close()
            logger.info("图RAG检索系统已关闭") 