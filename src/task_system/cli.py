"""Explicit task administration, also useful while both models are offline."""
import argparse
import json

from .client import TaskRPC


def main():
    parser = argparse.ArgumentParser()
    subs = parser.add_subparsers(dest='command', required=True)
    for name in ('list', 'runs'):
        subs.add_parser(name)
    for name in ('start', 'stop', 'delete', 'run'):
        subs.add_parser(name).add_argument('task_id', type=int)
    p = subs.add_parser('period')
    p.add_argument('task_id', type=int)
    p.add_argument('seconds', type=float)
    p = subs.add_parser('executor')
    p.add_argument('task_id', type=int)
    p.add_argument('executor', choices=('litert', 'openai'))
    p = subs.add_parser('create')
    p.add_argument('--description', required=True)
    p.add_argument('--text', required=True)
    p.add_argument('--skill', action='append', required=True, dest='skills')
    p.add_argument('--period', type=float, required=True)
    p.add_argument('--executor', choices=('litert', 'openai'), default='litert')
    args = parser.parse_args()
    rpc = TaskRPC()
    op = args.command
    if op == 'list':
        result = rpc.call('system', 'task_status_text')
    elif op == 'runs':
        result = rpc.call('system', 'runs')
    elif op == 'create':
        result = rpc.call('system', 'create_periodic_task', args.description, args.text,
                          args.skills, args.period, method='query', executor=args.executor)
    elif op == 'executor':
        result = rpc.call('tasks', 'set_executor', args.task_id, args.executor)
    elif op == 'period':
        result = rpc.call('system', 'set_task_period', args.task_id, args.seconds)
    elif op == 'run':
        result = rpc.call('system', 'run_task', args.task_id)
    else:
        result = rpc.call('system', op + '_task', args.task_id)
    print(result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
