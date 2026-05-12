import { useEffect, useRef } from 'react'
import { io, Socket } from 'socket.io-client'
import { useQueryClient } from '@tanstack/react-query'

type EventHandler = (data: any) => void

interface WebSocketContext {
  socket: Socket | null
  subscribe: (event: string, handler: EventHandler) => () => void
}

let globalSocket: Socket | null = null
const listeners = new Map<string, Set<EventHandler>>()

function getOrCreateSocket(): Socket {
  if (globalSocket && globalSocket.connected) return globalSocket
  globalSocket = io('/', {
    path: '/socket.io',
    transports: ['websocket', 'polling'],
    reconnection: true,
    reconnectionDelay: 1000,
    reconnectionDelayMax: 30000,
  })
  return globalSocket
}

export function useWebSocket(): WebSocketContext {
  const qc = useQueryClient()
  const socketRef = useRef<Socket | null>(null)

  useEffect(() => {
    const socket = getOrCreateSocket()
    socketRef.current = socket

    const onProjectUpdate = (data: { project: string; state: any }) => {
      qc.invalidateQueries({ queryKey: ['projects'] })
      qc.invalidateQueries({ queryKey: ['img2vid'] })
      const handlers = listeners.get('project_update')
      if (handlers) handlers.forEach((h) => h(data))
    }

    const onOptimizeProgress = (data: any) => {
      const handlers = listeners.get('optimize_progress')
      if (handlers) handlers.forEach((h) => h(data))
    }

    const onMatchProgress = (data: any) => {
      const handlers = listeners.get('match_progress')
      if (handlers) handlers.forEach((h) => h(data))
    }

    socket.on('project_update', onProjectUpdate)
    socket.on('optimize_progress', onOptimizeProgress)
    socket.on('match_progress', onMatchProgress)

    return () => {
      socket.off('project_update', onProjectUpdate)
      socket.off('optimize_progress', onOptimizeProgress)
      socket.off('match_progress', onMatchProgress)
    }
  }, [qc])

  const subscribe = (event: string, handler: EventHandler) => {
    if (!listeners.has(event)) listeners.set(event, new Set())
    listeners.get(event)!.add(handler)
    return () => {
      listeners.get(event)?.delete(handler)
    }
  }

  return { socket: socketRef.current, subscribe }
}
