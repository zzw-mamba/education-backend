"""生成日志路由模块

提供用户生成历史记录的查询功能，支持分页和按时间排序。
"""

from datetime import datetime
from typing import List

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from database import get_db
from models import Log, Template, User
from routers.user import get_current_user

router = APIRouter(prefix="/log", tags=["Log"])


def _parse_knowledge_ids(raw_ids: str | None) -> list[int]:
	"""将 logs.knowledge_ids 的逗号分隔字符串解析为整数列表。"""
	if not raw_ids:
		return []

	parsed: list[int] = []
	for item in raw_ids.split(","):
		item = item.strip()
		if not item:
			continue
		try:
			parsed.append(int(item))
		except ValueError:
			continue
	return parsed


@router.get("/my", status_code=200)
def get_my_generation_logs(
	page: int = 1,
	page_size: int = 20,
	current_user: User = Depends(get_current_user),
	db: Session = Depends(get_db),
):
	"""通过 token 获取当前用户的生成历史（按时间倒序）。"""
	page = max(page, 1)
	page_size = min(max(page_size, 1), 100)
	offset = (page - 1) * page_size

	base_query = db.query(Log).filter(
		Log.user_id == current_user.id,
		Log.template_id.isnot(None),
	)
	total = base_query.count()

	logs = (
		base_query.order_by(Log.created_at.desc())
		.offset(offset)
		.limit(page_size)
		.all()
	)

	template_ids = [item.template_id for item in logs if item.template_id is not None]
	template_map = {
		template.id: template.name
		for template in db.query(Template).filter(Template.id.in_(template_ids)).all()
	} if template_ids else {}

	data = []
	for item in logs:
		created_at: datetime | None = item.created_at
		data.append(
			{
				"log_id": item.id,
				"template_id": item.template_id,
				"template_name": template_map.get(item.template_id),
				"knowledge_ids": _parse_knowledge_ids(item.knowledge_ids),
				"knowledge_ids_raw": item.knowledge_ids,
				"result_path": item.result_path,
				"created_at": created_at.isoformat() if created_at else None,
			}
		)
	print(data)
	return {
		"code": 200,
		"message": "获取生成历史成功",
		"data": data,
		"pagination": {
			"page": page,
			"page_size": page_size,
			"total": total,
		},
	}
