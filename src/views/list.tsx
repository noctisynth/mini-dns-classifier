import Header from '@/components/Header';
import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router';
import * as api from '@/lib/api';
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import { BaggageClaimIcon, LinkIcon } from 'lucide-react';
import { Skeleton } from '@/components/ui/skeleton';

export default function List() {
  const redirect = useNavigate();

  const [loaded, setLoaded] = useState<boolean>(false);
  const [reports, setReports] = useState<
    {
      id: string;
      report: api.AnalysisReport;
      created_at: string;
    }[]
  >([]);
  useEffect(() => {
    api
      .getReports()
      .then(({ data }) => {
        console.log(data);
        setReports(data);
      })
      .finally(() => {
        setLoaded(true);
      });
  }, []);
  return (
    <>
      <div className="bg-background text-foreground min-h-screen">
        <Header />
        <main className="flex h-full w-full items-center justify-center p-4">
          <div className="flex flex-col gap-3 max-w-[756px] w-full">
            {!loaded && (
              <>
                <Skeleton className="w-full h-30" />
                <Skeleton className="w-full h-30" />
                <Skeleton className="w-full h-30" />
              </>
            )}
            {loaded &&
              reports.map((record, idx) => {
                return (
                  <Card
                    className="w-full hover:shadow-2xl hover:cursor-pointer"
                    key={`${idx}_${record.created_at}`}
                    onClick={() => redirect(`/report/${record.id}`)}
                  >
                    <CardHeader>
                      <CardTitle>报告 {record.id}</CardTitle>
                      <CardDescription>
                        {new Date(record.created_at).toLocaleString()}
                      </CardDescription>
                    </CardHeader>
                    <CardContent>
                      <div className="flex flex-row flex-wrap gap-3">
                        <div className="flex flex-row gap-1 items-center">
                          <BaggageClaimIcon className="h-3 w-3" />
                          <span className="text-sm">
                            总数据包：{record.report.total_packets}
                          </span>
                        </div>
                        <div className="flex flex-row gap-1 items-center">
                          <LinkIcon className="h-3 w-3" />
                          <span className="text-sm">
                            隐蔽信道：{record.report.covert_count}
                          </span>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                );
              })}
          </div>
        </main>
        <footer className="text-center text-sm text-gray-500 p-4">
        © 2025 DNS 分类系统
      </footer>
      </div>
    </>
  );
}
