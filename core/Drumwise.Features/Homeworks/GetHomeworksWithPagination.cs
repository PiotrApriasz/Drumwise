using AutoMapper;
using AutoMapper.QueryableExtensions;
using Drumwise.Application.Common.Errors;
using Drumwise.Application.Common.Interfaces;
using Drumwise.Application.Common.Mappings;
using Drumwise.Application.Common.Models;
using Drumwise.Application.Entities;
using Drumwise.Infrastructure.Data;
using Drumwise.Infrastructure.Identity.Constants;
using MediatR;

namespace Drumwise.Features.Homeworks;

public record GetHomeworksWithPaginationQuery(int PageNumber, int PageSize) 
    : IRequest<(Result, PaginatedList<HomeworkItemBriefDto>)>;

public class GetHomeworksWithPaginationHandler(ApplicationDbContext context, IMapper mapper, IUser user) 
    : IRequestHandler<GetHomeworksWithPaginationQuery, (Result, PaginatedList<HomeworkItemBriefDto>)>
{
    public async Task<(Result, PaginatedList<HomeworkItemBriefDto>)> Handle(GetHomeworksWithPaginationQuery request, CancellationToken cancellationToken)
    {
        var loggedUser = user.Id;
        var ifTeacher = await user.IsInRoleAsync(Roles.Teacher);
        
        var homeworksRaw = ifTeacher
            ? context.Homeworks.Where(x => x.CreatedBy == loggedUser)
            : context.Homeworks.Where(x => x.AssignedTo == loggedUser);
            
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