using AutoMapper;
using AutoMapper.QueryableExtensions;
using Drumwise.Application.Common.Errors;
using Drumwise.Application.Common.Mappings;
using Drumwise.Application.Common.Models;
using Drumwise.Application.Entities;
using Drumwise.Infrastructure.Data;
using MediatR;

namespace Drumwise.Features.Homeworks;

public record GetHomeworksWithPaginationQuery(
    string UserId,
    bool IfTeacher,
    int PageNumber,
    int PageSize) : IRequest<(Result, PaginatedList<HomeworkItemBriefDto>)>;

public class GetHomeworksWithPaginationHandler(ApplicationDbContext context, IMapper mapper) 
    : IRequestHandler<GetHomeworksWithPaginationQuery, (Result, PaginatedList<HomeworkItemBriefDto>)>
{
    public async Task<(Result, PaginatedList<HomeworkItemBriefDto>)> Handle(GetHomeworksWithPaginationQuery request, CancellationToken cancellationToken)
    {
        var homeworksRaw = request.IfTeacher
            ? context.Homeworks.Where(x => x.CreatedBy == request.UserId)
            : context.Homeworks.Where(x => x.AssignedTo == request.UserId);
            
        var homeworks = await homeworksRaw
            .OrderByDescending(x => x.Deadline)
            .ProjectTo<HomeworkItemBriefDto>(mapper.ConfigurationProvider)
            .PaginatedListAsync(request.PageNumber, request.PageSize)
            .ConfigureAwait(false);

        return (Result.Success(ResultType.Ok), homeworks);
    }
}

public record HomeworkItemBriefDto : IMapFrom<Homework>
{
    public Guid Id { get; set; }
    public DateTime Deadline { get; set; }
    public required string HomeworkTitle { get; set; }
    public required string AssignedTo { get; set; }
    public required string CreatedBy { get; set; }
}